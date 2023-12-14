import time
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import factorized

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from basix.ufl import element
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator

from multi.domain import Domain
from multi.bcs import get_boundary_dofs
from multi.boundary import within_range
from multi.materials import LinearElasticMaterial
from multi.product import InnerProduct
from multi.projection import orthogonal_part
from multi.problems import LinearElasticityProblem
from multi.solver import build_nullspace


def transfer_operator_subdomains_2d(A, dirichlet_dofs, target_dofs, projection_matrix=None):
    # FIXME what format (csr, csc) should A have for efficient slicing?

    # dirichlet dofs associated with Γ_out
    num_dofs = A.shape[0]
    all_dofs = np.arange(num_dofs)
    all_inner_dofs = np.setdiff1d(all_dofs, dirichlet_dofs)

    full_operator = A.copy()
    operator = full_operator[:, all_inner_dofs][all_inner_dofs, :]

    # factorization
    matrix_shape = operator.shape
    start = time.time()
    operator = factorized(operator)
    end = time.time()
    print(f"factorization of {matrix_shape} matrix in {end-start}", flush=True)

    # mapping from old to new dof numbers
    newdofs = np.zeros((num_dofs,), dtype=int)
    newdofs[all_inner_dofs] = np.arange(all_inner_dofs.size)
    range_dofs = newdofs[target_dofs]

    rhs_op = full_operator[:, dirichlet_dofs][all_inner_dofs, :]
    start = time.time()
    transfer_operator = -operator(rhs_op.todense())[range_dofs, :]
    end = time.time()
    print(f"applied operator to rhs in {end-start}", flush=True)

    if projection_matrix is not None:
        # remove kernel
        assert transfer_operator.shape[0] == projection_matrix.shape[0]
        P = np.eye(transfer_operator.shape[0]) - projection_matrix
        transfer_operator = P.dot(transfer_operator)

    return NumpyMatrixOperator(transfer_operator)


if __name__ == "__main__":
    # create oversampling domain 3x3 coarse cells
    # each coarse cell is discretized with 2x2 quads
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 6, 6, mesh.CellType.quadrilateral)
    # discretize A
    fe_deg = 1
    fe = element("P", domain.basix_cell(), fe_deg, shape=(2,))
    V = fem.functionspace(domain, fe)

    gdim = domain.ufl_cell().geometric_dimension()
    mat = LinearElasticMaterial(gdim, 30e3, 0.3, plane_stress=True)
    omega = Domain(domain)
    problem = LinearElasticityProblem(omega, V, (mat,))
    problem.setup_solver()
    problem.assemble_matrix()
    ai, aj, av = problem._A.getValuesCSR()
    # convert to other format? csc?
    A = csr_array((av, aj, ai))

    # dof indices
    boundary_dofs = get_boundary_dofs(V)
    tdim = domain.topology.dim
    target_subdomain = within_range([2./6, 2./6], [4./6, 4./6])
    range_cells = mesh.locate_entities(domain, tdim, target_subdomain)
    range_dofs = fem.locate_dofs_topological(V, tdim, range_cells)

    # FIXME V.dofmap.bs == 2 is not considered in boundary_dofs and range_dofs
    # workaround: create bc and get dof indices
    zero = fem.Constant(domain, (default_scalar_type(0.), ) * gdim)
    bc_boundary = fem.dirichletbc(zero, boundary_dofs, V)
    bc_target = fem.dirichletbc(zero, range_dofs, V)

    # build projection matrix in the range space!
    # P = R inv( R^T R ) R^T, with R holding rigid body modes as column vectors
    # TODO build submesh ... then space on target subdomain
    tdim = domain.topology.dim
    cells_in = mesh.locate_entities(domain, tdim, target_subdomain)
    omega_in, _, _, _= mesh.create_submesh(domain, tdim, cells_in)
    W = fem.functionspace(omega_in, fe)
    rangespace = FenicsxVectorSpace(W)

    # select inner product
    # inner_product = "eulidean"
    inner_product = "h1"
    # inner_product = "mass"

    product = None
    M = np.eye(rangespace.dim)
    if not inner_product == "eulidean":
        product = InnerProduct(W, product=inner_product)
        product_mat = product.assemble_matrix()
        product = FenicsxMatrixOperator(product_mat, W, W)
        M = product_mat[:, :]

    basis = build_nullspace(rangespace, product=product)
    R = basis.to_numpy().T

    right = np.dot(R.T, M)
    middle = np.linalg.inv(np.dot(R.T, np.dot(M, R)))
    left = R
    P = np.dot(left, np.dot(middle, right))

    random_values = rangespace.random(1)
    coeff = np.array([1.3, 4.9, 0.6])
    # coeff.shape[-1] has to agree with len(basis)
    U = basis.lincomb(coeff) + random_values

    # ### compute orthogonal part
    # 1. via matrix
    r1 = (np.eye(P.shape[0]) - P).dot(U.to_numpy().flatten())

    # 2. solving linear system
    G = basis.gramian(product=product)
    rhs = basis.inner(U, product=product)
    v = np.linalg.solve(G, rhs)
    U_proj = basis.lincomb(v.T)
    r2 = U - U_proj
    # r1 and r2 are equal
    # but they do not match random values
    # the projection will try to find closest match
    # but will make an error since its not possible to represent U in basis
    error = r1 - r2.to_numpy().flatten()
    assert np.linalg.norm(error) < 1e-12

    # build transfer operator
    boundary_dofs_full = bc_boundary._cpp_object.dof_indices()[0]
    range_dofs_full = bc_target._cpp_object.dof_indices()[0]
    T = transfer_operator_subdomains_2d(A, boundary_dofs_full, range_dofs_full, projection_matrix=None)
    # compare T.apply to A.apply_inverse restricted to target subdomain
    U = T.source.random(1)
    TU = T.apply(U)

    Tproj = transfer_operator_subdomains_2d(A, boundary_dofs_full, range_dofs_full, projection_matrix=P)
    TpU = Tproj.apply(U)

    # reference solution
    g = fem.Function(V)
    g.vector.array[boundary_dofs_full] = U.to_numpy().flatten()
    bc = fem.dirichletbc(g, boundary_dofs)
    problem.clear_bcs()
    problem.add_dirichlet_bc(bc)
    problem.setup_solver()
    u = problem.solve()
    u_in = u.vector.array[range_dofs_full]

    # comparison without removing the kernel
    error = u_in - TU.to_numpy().flatten()
    assert np.linalg.norm(error) < 1e-9

    # comparison with removing the kernel
    u_in_orth = (np.eye(P.shape[0]) - P).dot(u_in)
    error = u_in_orth - TpU.to_numpy().flatten()
    assert np.linalg.norm(error) < 1e-9

    # test against alternative orthogonal part computation
    UIN = rangespace.from_numpy([u_in])
    U_orth = orthogonal_part(basis, UIN, product, orth=False)
    error = U_orth.to_numpy().flatten() - TpU.to_numpy().flatten()
    assert np.linalg.norm(error) < 1e-9
