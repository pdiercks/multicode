import numpy as np
from scipy.sparse import csr_array

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from basix.ufl import element
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from multi.domain import Domain
from multi.bcs import get_boundary_dofs
from multi.boundary import within_range, plane_at
from multi.materials import LinearElasticMaterial
from multi.product import InnerProduct
from multi.projection import orthogonal_part
from multi.problems import LinearElasticityProblem
from multi.solver import build_nullspace
from multi.transfer_operator import transfer_operator_subdomains_2d, OrthogonallyProjectedOperator
import pytest


@pytest.mark.parametrize("product_name",["euclidean","h1"])
def test(product_name):
    """test transfer operator with ∂Ω=Γ_out"""

    # create oversampling domain 3x3 coarse cells
    # each coarse cell is discretized with resolutionxresolution quads
    num_cells = 3
    resolution = 10 # manually tested resolution = 50 with success
    nn = num_cells * resolution
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nn, nn, mesh.CellType.quadrilateral)
    # discretize A
    fe_deg = 1
    fe = element("P", domain.basix_cell(), fe_deg, shape=(2,))
    V = fem.functionspace(domain, fe)
    print(f"V dim: {V.dofmap.index_map.size_global * V.dofmap.bs}")

    gdim = domain.ufl_cell().geometric_dimension()
    mat = LinearElasticMaterial(gdim, 30e3, 0.3, plane_stress=True)
    omega = Domain(domain)
    problem = LinearElasticityProblem(omega, V, (mat,))
    problem.setup_solver()
    problem.assemble_matrix()
    ai, aj, av = problem._A.getValuesCSR()
    A = csr_array((av, aj, ai))

    # dof indices
    boundary_dofs = get_boundary_dofs(V)
    tdim = domain.topology.dim
    target_subdomain = within_range([1./num_cells, 1./num_cells], [2./num_cells, 2./num_cells])
    range_cells = mesh.locate_entities(domain, tdim, target_subdomain)
    range_dofs = fem.locate_dofs_topological(V, tdim, range_cells)

    # FIXME V.dofmap.bs == 2 is not considered in boundary_dofs and range_dofs
    # workaround: create bc and get dof indices
    zero = fem.Constant(domain, (default_scalar_type(0.), ) * gdim)
    bc_boundary = fem.dirichletbc(zero, boundary_dofs, V)
    bc_target = fem.dirichletbc(zero, range_dofs, V)

    # build projection matrix in the range space!
    tdim = domain.topology.dim
    cells_in = mesh.locate_entities(domain, tdim, target_subdomain)
    omega_in, _, _, _= mesh.create_submesh(domain, tdim, cells_in)
    W = fem.functionspace(omega_in, fe)
    rangespace = FenicsxVectorSpace(W)

    product = None
    product_numpy = None
    if not product_name == "euclidean":
        product = InnerProduct(W, product_name)
        product_mat = product.assemble_matrix()
        product = FenicsxMatrixOperator(product_mat, W, W)
        product_numpy = NumpyMatrixOperator(
                csr_array(product.matrix.getValuesCSR()[::-1])
                )

    nullspace = build_nullspace(rangespace, product=product)

    # build transfer operator
    boundary_dofs_full = bc_boundary._cpp_object.dof_indices()[0]
    range_dofs_full = bc_target._cpp_object.dof_indices()[0]

    # without removing kernel
    T_hat = transfer_operator_subdomains_2d(A, boundary_dofs_full, range_dofs_full)
    U = T_hat.source.random(1)
    ThatU = T_hat.apply(U)

    # with removing kernel
    # T_hat is a NumpyMatrixOperator and therefore nullspace is not in T_hat.range
    # FIXME: this conversion should be avoided
    basis = T_hat.range.from_numpy(nullspace.to_numpy())
    Tproj = OrthogonallyProjectedOperator(T_hat, basis, product=product_numpy, orthonormal=True)
    ops = [T_hat, Tproj]
    coeffs = [1., -1.]
    T = LincombOperator(ops, coeffs)
    TpU = T.apply(U)


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
    error = u_in - ThatU.to_numpy().flatten()
    assert np.linalg.norm(error).item() < 1e-9

    # test against alternative orthogonal part computation
    UIN = rangespace.from_numpy([u_in])
    U_orth = orthogonal_part(UIN, nullspace, product=product, orthonormal=True)
    error = U_orth.to_numpy().flatten() - TpU.to_numpy().flatten()
    assert np.linalg.norm(error).item() < 1e-9


@pytest.mark.parametrize("product_name",["euclidean","h1"])
def test_bc_hom(product_name):
    """test transfer operator with homogeneous Dirichlet BCs on some part of ∂Ω"""
    # create oversampling domain 2x2 coarse cells (e. g. corner case)
    # corner case usually means we have the following topology:
    # Σ_D = left, Σ_N = bottom, Γ_out = remainder (within structure)
    # each coarse cell is discretized with 2x2 quads
    num_cells = 2
    resolution = 6
    nn = num_cells * resolution
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nn, nn, mesh.CellType.quadrilateral)
    fe_deg = 1
    fe = element("P", domain.basix_cell(), fe_deg, shape=(2,))
    V = fem.functionspace(domain, fe)

    # ### Define homogeneous Dirichlet BC
    sigma_D = plane_at(0., "x")
    tdim = domain.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, sigma_D)
    gdim = domain.ufl_cell().geometric_dimension()
    zero = fem.Constant(domain, (default_scalar_type(0.), ) * gdim)

    mat = LinearElasticMaterial(gdim, 30e3, 0.3, plane_stress=True)
    omega = Domain(domain)
    problem = LinearElasticityProblem(omega, V, (mat,))
    problem.add_dirichlet_bc(zero, facets, entity_dim=fdim)
    problem.setup_solver()
    problem.assemble_matrix(bcs=problem.bcs)
    ai, aj, av = problem._A.getValuesCSR()
    A = csr_array((av, aj, ai))

    # ### prepare data for Γ_out and range space on target subdomain
    # Γ_out should be union of top and right boundary excl. points on Σ_D or Σ_N
    # simpley exclude left and bottom boundary, but mark everything else
    Δx = Δy = 1e-2 # should be smaller than local mesh cell size
    gamma_out = within_range([Δx, Δy], [1., 1.])
    boundary_dofs = get_boundary_dofs(V, marker=gamma_out)
    target_subdomain = within_range([0., 0.], [1./num_cells, 1./num_cells])
    range_cells = mesh.locate_entities(domain, tdim, target_subdomain)
    range_dofs = fem.locate_dofs_topological(V, tdim, range_cells)

    # FIXME V.dofmap.bs == 2 is not considered in boundary_dofs and range_dofs
    # workaround: create bc and get dof indices
    bc_boundary = fem.dirichletbc(zero, boundary_dofs, V)
    bc_target = fem.dirichletbc(zero, range_dofs, V)

    # build projection matrix in the range space!
    tdim = domain.topology.dim
    cells_in = mesh.locate_entities(domain, tdim, target_subdomain)
    omega_in, _, _, _= mesh.create_submesh(domain, tdim, cells_in)
    W = fem.functionspace(omega_in, fe)
    rangespace = FenicsxVectorSpace(W)

    product = None
    if not product_name == "euclidean":
        product = InnerProduct(W, product_name)
        product_mat = product.assemble_matrix()
        product = FenicsxMatrixOperator(product_mat, W, W)

    basis = build_nullspace(rangespace, product=product)

    _dofs_sigma_d = get_boundary_dofs(V, marker=sigma_D)
    bc_hom = fem.dirichletbc(zero, _dofs_sigma_d, V)
    V_dofs_sigma_d = bc_hom._cpp_object.dof_indices()[0]
    _dofs_sigma_d = get_boundary_dofs(W, marker=sigma_D)
    bc_hom = fem.dirichletbc(zero, _dofs_sigma_d, W)
    W_dofs_sigma_d = bc_hom._cpp_object.dof_indices()[0]

    # build transfer operator
    boundary_dofs_full = bc_boundary._cpp_object.dof_indices()[0]
    range_dofs_full = bc_target._cpp_object.dof_indices()[0]
    # projection to remove kernel should not be necessary
    T = transfer_operator_subdomains_2d(A, boundary_dofs_full, range_dofs_full)
    U = T.source.random(1)
    TU = T.apply(U)

    # ### reference solution
    g = fem.Function(V)
    g.vector.array[boundary_dofs_full] = U.to_numpy().flatten()
    bc = fem.dirichletbc(g, boundary_dofs)
    # so far only added homogeneous Dirichlet bc
    problem.add_dirichlet_bc(bc) # add some random values on Γ_out
    problem.setup_solver()
    u = problem.solve()
    u_in = u.vector.array[range_dofs_full]

    # ### check homogeneous Dirichlet bc is satisfied
    assert np.isclose(np.sum(u.vector.array[V_dofs_sigma_d]), 0.0)
    assert np.isclose(np.sum(TU.dofs(W_dofs_sigma_d)), 0.0)

    # comparison with reference fem solution
    error = u_in - TU.to_numpy().flatten()
    assert np.linalg.norm(error).item() < 1e-9

    # check that u is not in ker(A)
    # if u in ker(A) then the projection error should be zero
    UIN = rangespace.from_numpy([u_in])
    Uorth = orthogonal_part(UIN, basis, product=product, orthonormal=True)
    assert Uorth.norm(product).item() > 0
    
    space = NumpyVectorSpace(3)
    alpha = space.random(3)
    va = basis.lincomb(alpha.to_numpy())
    va_orth = orthogonal_part(va, basis, product=product, orthonormal=True)
    assert np.allclose(va_orth.norm(product), np.zeros(3))


if __name__ == "__main__":
    test("h1")
    test_bc_hom("h1")
