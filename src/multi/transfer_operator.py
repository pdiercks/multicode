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
# from multi.misc import x_dofs_vectorspace
from multi.product import InnerProduct
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.solver import build_nullspace


def transfer_operator_subdomains_2d(A, dirichlet_dofs, target_dofs):

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

    # TODO projection | remove kernel
    # TODO compute projection matrix P as R inv( R^T R ) R^T, with R holding rigid body modes as column vectors
    # if ar:
    #     ar = average_remover(transfer_operator.shape[0])
    #     transfer_operator = ar.dot(transfer_operator)

    return NumpyMatrixOperator(transfer_operator)


def test_matrix_of_projection():
    # assemble matrix of rigid body modes --> R
    # orthonormalize
    # compute R inv( R^T R ) R^T
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2, mesh.CellType.quadrilateral)
    fe_deg = 1
    fe = element("P", domain.basix_cell(), fe_deg, shape=(2,))
    V = fem.functionspace(domain, fe)

    source = FenicsxVectorSpace(V)
    product = InnerProduct(V, product="h1")
    product_mat = product.assemble_matrix()
    product = FenicsxMatrixOperator(product_mat, V, V)
    ns = build_nullspace(source, product=product)

    u = fem.Function(V)
    gamma = 0.2 # simple shear factor
    u.interpolate(lambda x: [0.5 + x[1] * gamma, 0.3 + x[0] * 0.]) # type: ignore
    U = source.make_array([u.vector]) # type: ignore

    # projection
    coeff = U.inner(ns, product)
    U_proj = ns.lincomb(coeff)

    R = ns.to_numpy()
    A = R.T
    P = np.dot(A, np.dot(np.linalg.inv(np.dot(A.T, A)), A.T))
    result = np.dot(P, U.to_numpy().flatten())
    assert np.allclose(result, U_proj.to_numpy().flatten())


if __name__ == "__main__":
    test_matrix_of_projection()

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

    # build transfer operator
    boundary_dofs_full = bc_boundary._cpp_object.dof_indices()[0]
    range_dofs_full = bc_target._cpp_object.dof_indices()[0]
    T = transfer_operator_subdomains_2d(A, boundary_dofs_full, range_dofs_full)
    # compare T.apply to A.apply_inverse restricted to target subdomain
    U = T.source.random(1)
    TU = T.apply(U)

    # reference solution
    g = fem.Function(V)
    g.vector.array[boundary_dofs_full] = U.to_numpy().flatten()
    bc = fem.dirichletbc(g, boundary_dofs)

    problem.clear_bcs()
    problem.add_dirichlet_bc(bc)
    problem.setup_solver()
    u = problem.solve()
    u_in = u.vector.array[range_dofs_full]
    error = u_in - TU.to_numpy().flatten()
    assert np.linalg.norm(error) < 1e-9
