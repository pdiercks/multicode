import tempfile
import numpy as np

from mpi4py import MPI
import dolfinx as df
import basix
import basix.ufl

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenicsx import (
    FenicsxVectorSpace,
    FenicsxMatrixOperator,
)
from pymor.operators.constructions import LincombOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from multi.preprocessing import create_rectangle
from multi.domain import Domain
from multi.bcs import get_boundary_dofs
from multi.boundary import within_range, plane_at
from multi.materials import LinearElasticMaterial
from multi.product import InnerProduct
from multi.projection import orthogonal_part
from multi.problems import LinearElasticityProblem
from multi.solver import build_nullspace
from multi.transfer_operator import (
    discretize_transfer_operator,
    OrthogonallyProjectedOperator,
)
from multi.misc import x_dofs_vectorspace
from multi.interpolation import make_mapping
import pytest


@pytest.mark.parametrize("product_name", ["euclidean", "h1"])
def test(product_name):
    """test transfer operator with ∂Ω=Γ_out"""

    # ### Oversampling domain
    gdim = 2
    ul = 100.0
    num_cells = 3
    resolution = 20
    n = num_cells * resolution
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(
            0.0,
            num_cells * ul,
            0.0,
            num_cells * ul,
            num_cells=(n, n),
            facet_tags={"bottom": 1, "left": 2, "right": 3, "top": 4},
            recombine=True,
            out_file=tf.name,
        )
        omega, _, facet_markers = df.io.gmshio.read_from_msh(
            tf.name, MPI.COMM_WORLD, gdim=gdim
        )

    # ### Finite elements and FE space
    fe_deg = 1
    gdim = omega.geometry.dim
    fe_quad = basix.ufl.element("P", omega.basix_cell(), fe_deg, shape=(gdim,))
    fe_line = basix.ufl.element("P", basix.CellType.interval, fe_deg, shape=(gdim,))
    V = df.fem.functionspace(omega, fe_quad)

    # ### Full operator A
    mat = LinearElasticMaterial(gdim, 30e3, 0.2, plane_stress=False)
    Omega = Domain(omega)
    problem = LinearElasticityProblem(Omega, V, mat)
    problem.setup_solver()
    problem.assemble_matrix()
    A = problem.A

    # ### Target subdomain
    tdim = omega.topology.dim
    omega.topology.create_connectivity(tdim, tdim)
    marker_omega_in = within_range([ul, ul, 0], [2 * ul, 2 * ul, 0])
    cells_subdomain = df.mesh.locate_entities(omega, tdim, marker_omega_in)
    submesh = df.mesh.create_submesh(omega, tdim, cells_subdomain)[0]
    assert np.isclose(cells_subdomain.size, resolution**2)

    # ### Gamma out (the whole boundary)
    def everywhere(x):
        return np.full(x[0].shape, True, dtype=bool)

    facets_gamma_out = df.mesh.locate_entities_boundary(omega, tdim - 1, everywhere)
    gamma_out = df.mesh.create_submesh(omega, tdim - 1, facets_gamma_out)[0]
    assert np.isclose(facets_gamma_out.size, num_cells * resolution * 4)
    gamma_dofs_ = df.fem.locate_dofs_topological(V, tdim - 1, facets_gamma_out)

    # NOTE
    # We could use `locate_dofs_topological` to determine `gamma_dofs`.
    # However, since `FenicsxMatrixOperator` requires a source space
    # the "link" between `gamma_dofs` and `source` would be lost.
    # (the dof ordering would be wrong)
    # Therefore, `gamma_dofs` & `target_dofs` are determined via `make_mapping`.

    # ### source and range space of T
    S = df.fem.functionspace(gamma_out, fe_line)
    R = df.fem.functionspace(submesh, fe_quad)
    range_space = FenicsxVectorSpace(R)

    gamma_dofs = make_mapping(S, V, padding=1e-12, check=True)
    target_dofs = make_mapping(R, V, padding=1e-12, check=True)

    # compare ordering of dofs
    x_dofs_V = x_dofs_vectorspace(V)
    egamma = x_dofs_V[gamma_dofs] - x_dofs_vectorspace(S)
    etarget = x_dofs_V[target_dofs] - x_dofs_vectorspace(R)
    assert np.linalg.norm(egamma) < 1e-12
    assert np.linalg.norm(etarget) < 1e-12

    product = None
    if not product_name == "euclidean":
        product = InnerProduct(R, product_name)
        product_mat = product.assemble_matrix()
        product = FenicsxMatrixOperator(product_mat, R, R)

    ns_vecs = build_nullspace(R, gdim=2)
    nullspace = range_space.make_array(ns_vecs)
    gram_schmidt(nullspace, product=product, copy=False)

    # without removing kernel
    t_mat = discretize_transfer_operator(A, gamma_dofs, target_dofs)
    T_hat = FenicsxMatrixOperator(t_mat, S, R)
    U = T_hat.source.random(1, distribution="normal")
    ThatU = T_hat.apply(U)

    # with removing kernel
    Tproj = OrthogonallyProjectedOperator(
        T_hat, nullspace, product=product, orthonormal=True
    )
    ops = [T_hat, Tproj]
    coeffs = [1.0, -1.0]
    T = LincombOperator(ops, coeffs)
    TpU = T.apply(U)

    # reference solution
    g = df.fem.Function(V)
    g.vector.array[gamma_dofs] = U.to_numpy().flatten()
    bc = df.fem.dirichletbc(g, gamma_dofs_)
    problem.clear_bcs()
    problem.add_dirichlet_bc(bc)
    problem.setup_solver()
    u = problem.solve()
    u_in = u.vector.array[target_dofs]

    # comparison without removing the kernel
    error = u_in - ThatU.to_numpy().flatten()
    assert np.linalg.norm(error).item() < 1e-9

    # test against alternative orthogonal part computation
    UIN = range_space.from_numpy([u_in])
    U_orth = orthogonal_part(UIN, nullspace, product=product, orthonormal=True)
    error = U_orth.to_numpy().flatten() - TpU.to_numpy().flatten()
    assert np.linalg.norm(error).item() < 1e-9


@pytest.mark.parametrize("product_name", ["euclidean", "h1"])
def test_bc_hom(product_name):
    """test transfer operator with homogeneous Dirichlet BCs on some part of ∂Ω"""

    # ### Oversampling domain
    # create oversampling domain 2x2 coarse cells (e. g. corner case)
    # corner case usually means we have the following topology:
    # Σ_D = left, Σ_N = bottom, Γ_out = remainder (within structure)
    # each coarse cell is discretized with 2x2 quads
    num_cells = 2
    resolution = 6
    nn = num_cells * resolution
    domain = df.mesh.create_unit_square(
        MPI.COMM_WORLD, nn, nn, df.mesh.CellType.quadrilateral
    )
    fe_deg = 1
    gdim = domain.geometry.dim
    fe_quad = basix.ufl.element("P", domain.basix_cell(), fe_deg, shape=(gdim,))
    fe_line = basix.ufl.element("P", basix.CellType.interval, fe_deg, shape=(gdim,))
    V = df.fem.functionspace(domain, fe_quad)

    # ### Define homogeneous Dirichlet BC
    sigma_D = plane_at(0.0, "x")
    tdim = domain.topology.dim
    fdim = tdim - 1
    facets_sigma_D = df.mesh.locate_entities_boundary(domain, fdim, sigma_D)
    gdim = domain.geometry.dim
    zero = df.fem.Constant(domain, (df.default_scalar_type(0.0),) * gdim)

    # ### Full operator A
    mat = LinearElasticMaterial(gdim, 30e3, 0.3, plane_stress=True)
    omega = Domain(domain)
    problem = LinearElasticityProblem(omega, V, mat)
    problem.add_dirichlet_bc(zero, facets_sigma_D, entity_dim=fdim)
    problem.setup_solver()
    problem.assemble_matrix(bcs=problem.bcs)
    A = problem._A

    # ### Target subdomain
    target_subdomain_locator = within_range(
        [0.0, 0.0], [1.0 / num_cells, 1.0 / num_cells]
    )
    domain.topology.create_connectivity(tdim, tdim)
    cells_subdomain = df.mesh.locate_entities(domain, tdim, target_subdomain_locator)
    submesh = df.mesh.create_submesh(domain, tdim, cells_subdomain)[0]
    assert np.isclose(cells_subdomain.size, resolution**2)

    # ### Gamma out
    # Γ_out should be union of top and right boundary excl. points on Σ_D or Σ_N
    # simply exclude left and bottom boundary, but mark everything else
    Δx = Δy = 1e-2  # should be smaller than local mesh cell size
    gamma_out_locator = within_range([Δx, Δy], [1.0, 1.0])
    facets_gamma_out = df.mesh.locate_entities_boundary(
        domain, tdim - 1, gamma_out_locator
    )
    gamma_out = df.mesh.create_submesh(domain, tdim - 1, facets_gamma_out)[0]
    assert np.isclose(facets_gamma_out.size, num_cells * resolution * 2 - 2)

    # ### source and range space of T
    S = df.fem.functionspace(gamma_out, fe_line)
    R = df.fem.functionspace(submesh, fe_quad)
    range_space = FenicsxVectorSpace(R)

    gamma_dofs = make_mapping(S, V, padding=1e-12, check=True)
    target_dofs = make_mapping(R, V, padding=1e-12, check=True)

    product = None
    if not product_name == "euclidean":
        product = InnerProduct(R, product_name)
        product_mat = product.assemble_matrix()
        product = FenicsxMatrixOperator(product_mat, R, R)

    ns_vecs = build_nullspace(R, gdim=gdim)
    basis = range_space.make_array(ns_vecs)
    gram_schmidt(basis, product=product, copy=False)

    _dofs_sigma_d = get_boundary_dofs(V, marker=sigma_D)
    bc_hom = df.fem.dirichletbc(zero, _dofs_sigma_d, V)
    V_dofs_sigma_d = bc_hom._cpp_object.dof_indices()[0]
    _dofs_sigma_d = get_boundary_dofs(R, marker=sigma_D)
    bc_hom = df.fem.dirichletbc(zero, _dofs_sigma_d, R)
    W_dofs_sigma_d = bc_hom._cpp_object.dof_indices()[0]

    # ### Build transfer operator
    # projection to remove kernel should not be necessary
    t_mat = discretize_transfer_operator(A, gamma_dofs, target_dofs)
    T = FenicsxMatrixOperator(t_mat, S, R)
    U = T.source.random(1)
    TU = T.apply(U)

    # ### reference solution
    g = df.fem.Function(V)
    g.x.array[gamma_dofs] = U.to_numpy().flatten()

    problem.clear_bcs()
    problem.add_dirichlet_bc(zero, facets_sigma_D, entity_dim=fdim)
    problem.add_dirichlet_bc(g, facets_gamma_out, entity_dim=fdim)
    problem.setup_solver()
    u = problem.solve()
    u.x.scatter_forward()
    u_in = u.x.array[target_dofs]

    # ### check homogeneous Dirichlet bc is satisfied
    assert np.isclose(np.sum(u.x.array[V_dofs_sigma_d]), 0.0)
    assert np.isclose(np.sum(TU.dofs(W_dofs_sigma_d)), 0.0)

    # comparison with reference fem solution
    error = u_in - TU.to_numpy().flatten()
    assert np.linalg.norm(error).item() < 1e-9

    # check that u is not in ker(A)
    # if u in ker(A) then the projection error should be zero
    UIN = range_space.from_numpy([u_in])
    Uorth = orthogonal_part(UIN, basis, product=product, orthonormal=True)
    assert Uorth.norm(product).item() > 0

    space = NumpyVectorSpace(3)
    alpha = space.random(3)
    va = basis.lincomb(alpha.to_numpy())
    va_orth = orthogonal_part(va, basis, product=product, orthonormal=True)
    assert np.allclose(va_orth.norm(product), np.zeros(3))


if __name__ == "__main__":
    test("euclidean")
    # test_bc_hom("h1")
