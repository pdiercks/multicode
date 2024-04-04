"""test oversampling problem discretization for
inhomogeneous neumann boundary conditions"""

from mpi4py import MPI
import tempfile
import numpy as np
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.io import gmshio
from basix.ufl import element
from multi.domain import RectangularSubdomain, RectangularDomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem, TransferProblem, LinElaSubProblem
from multi.boundary import plane_at, within_range, point_at
from multi.preprocessing import create_rectangle
from multi.projection import orthogonal_part, project_array
from multi.solver import build_nullspace
from multi.product import InnerProduct
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from multi.misc import locate_dofs, x_dofs_vectorspace


def exact_solution(problem, dirichlet_bc, Vsub):
    """exact solution in full space"""
    problem.clear_bcs()

    if dirichlet_bc is not None:
        if isinstance(dirichlet_bc, list):
            for bc in dirichlet_bc:
                problem.add_dirichlet_bc(**bc)
        else:
            problem.add_dirichlet_bc(**dirichlet_bc)

    # ### exact solution full space
    u_exact = problem.solve()

    u_in = fem.Function(Vsub)
    u_in.interpolate(
        u_exact,
        nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
            u_in.function_space.mesh,
            u_in.function_space.element,
            u_exact.function_space.mesh,
        ),
    )

    # clean up
    problem.clear_bcs()
    return u_in.vector.array


def test_remove_rot():
    """Topology

    Ω = (0, 1) x (0, 1)
    Ω_in = (0.0, 0.0) x (0.5, 0.5)
    Γ_out = right plane
    Σ_D_hom = origin

    Translations are constrained, but rotation is free.
    """

    def target_subdomain(x):
        tol = 1e-4
        a = x[0] <= 0.5 + tol
        b = x[1] <= 0.5 + tol
        return np.logical_and(a, b)

    n = 20
    gdim = 2
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(
            0.0,
            1.0,
            0.0,
            1.0,
            num_cells=(n, n),
            facet_tags={"bottom": 1, "left": 2, "right": 3, "top": 4},
            recombine=True,
            out_file=tf.name,
        )
        square, _, facet_markers = gmshio.read_from_msh(
            tf.name, MPI.COMM_WORLD, gdim=gdim
        )
    degree = 1
    ve = element("P", square.basix_cell(), degree, shape=(2,))
    V = fem.functionspace(square, ve)

    domain = RectangularDomain(square, facet_tags=facet_markers)
    phases = LinearElasticMaterial(gdim, 210e3, 0.3, plane_stress=True)
    problem = LinearElasticityProblem(domain, V, phases)
    # subdomain problem
    cells_submesh = mesh.locate_entities(domain.grid, 2, target_subdomain)
    submesh = mesh.create_submesh(domain.grid, 2, cells_submesh)[0]

    # submesh has same cell type, reuse ve
    Vsub = fem.functionspace(submesh, ve)

    subdomain = RectangularSubdomain(99, submesh)
    subproblem = LinElaSubProblem(subdomain, Vsub, phases)

    zero = fem.Constant(square, (default_scalar_type(0.0), default_scalar_type(0.0)))
    gamma_out = plane_at(1.0, "x")  # right
    x_origin = np.array([[0., 0., 0.]])
    origin = point_at(x_origin[0])
    dirichlet_bc = {"boundary": origin, "value": zero, "method": "geometrical"}

    inner_range_product = InnerProduct(Vsub, "h1")
    range_product_mat = inner_range_product.assemble_matrix()
    range_product = FenicsxMatrixOperator(range_product_mat, Vsub, Vsub)

    ns_vecs = build_nullspace(Vsub, gdim=submesh.geometry.dim)
    range_space = FenicsxVectorSpace(Vsub)
    nullspace = range_space.make_array([ns_vecs[-1]])
    gram_schmidt(nullspace, product=range_product, copy=False)

    tp = TransferProblem(
        problem,
        subproblem,
        gamma_out,
        dirichlet=dirichlet_bc,
        range_product=range_product,
        kernel=nullspace,
    )

    # generate boundary data
    D = tp.generate_random_boundary_data(2, distribution="normal")
    assert np.isclose(tp.source_gamma_out.dim, D.shape[-1])
    U = tp.solve(D)
    assert np.isclose(U.dim, tp.S_to_R.size)
    u_arr = U.to_numpy()

    dofs_origin = locate_dofs(x_dofs_vectorspace(Vsub), x_origin)
    zero_dofs = U.dofs(dofs_origin)
    assert np.allclose(zero_dofs, np.zeros_like(zero_dofs))

    # compute reference solutions
    u_ex = np.zeros_like(u_arr)
    dof_indices = tp.bc_dofs_gamma_out
    for i, vector in enumerate(D):
        boundary_function = fem.Function(V)
        boundary_vector = boundary_function.vector
        boundary_vector.array[dof_indices] = vector

        bc_gamma_out = {
            "boundary": gamma_out,
            "value": boundary_function,
            "method": "geometrical",
        }
        u_exact = exact_solution(problem, [bc_gamma_out, dirichlet_bc], Vsub)
        u_ex[i, :] = u_exact

    # remove kernel of exact solution
    UEX = tp.range.from_numpy(u_ex)
    U_proj = orthogonal_part(
        UEX, tp.kernel, product=tp.range_product, orthonormal=True
    )

    u_ex = U_proj.to_numpy()
    error = u_ex - u_arr
    norm = np.linalg.norm(error)
    assert norm < 1e-12


def test_remove_trans_x_rot():
    """Topology

    Ω = (0, 2) x (0, 1)
    Ω_in = (1.0, 2.0) x (0.0, 0.1)
    Γ_out = left plane at x=0.0
    Σ_D_hom = bottom right corner x=2., y=0.

    Translation in y-direction at bottom right corner is constrained, but translation in x and rotation is free.
    """

    def target_subdomain(x):
        tol = 1e-4
        a = x[0] >= 1.0 - tol
        b = x[1] <= 1.0 + tol
        return np.logical_and(a, b)

    n = 20
    gdim = 2
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(
            0.0,
            2.0,
            0.0,
            1.0,
            num_cells=(2*n, n),
            facet_tags={"bottom": 1, "left": 2, "right": 3, "top": 4},
            recombine=True,
            out_file=tf.name,
            options={'Mesh.ElementOrder': 2},
        )
        square, _, facet_markers = gmshio.read_from_msh(
            tf.name, MPI.COMM_WORLD, gdim=gdim
        )
    degree = 2
    ve = element("P", square.basix_cell(), degree, shape=(2,))
    V = fem.functionspace(square, ve)

    domain = RectangularDomain(square, facet_tags=facet_markers)
    phases = LinearElasticMaterial(gdim, 210e3, 0.3, plane_stress=True)
    problem = LinearElasticityProblem(domain, V, phases)
    # subdomain problem
    cells_submesh = mesh.locate_entities(domain.grid, 2, target_subdomain)
    assert np.isclose(cells_submesh.size, 20 ** 2)
    submesh = mesh.create_submesh(domain.grid, 2, cells_submesh)[0]

    # submesh has same cell type, reuse ve
    Vsub = fem.functionspace(submesh, ve)

    subdomain = RectangularSubdomain(99, submesh)
    subproblem = LinElaSubProblem(subdomain, Vsub, phases)

    uy_zero = default_scalar_type(0.0)
    x_bottom_right = np.array([[2., 0., 0.]])
    bottom_right = point_at(x_bottom_right[0])
    dirichlet_bc = {"boundary": bottom_right, "value": uy_zero, "sub": 1, "entity_dim": 0, "method": "geometrical"}

    subproblem.add_dirichlet_bc(**dirichlet_bc)
    bc_hom = subproblem.get_dirichlet_bcs()

    inner_range_product = InnerProduct(Vsub, "h1", bcs=bc_hom)
    range_product_mat = inner_range_product.assemble_matrix()
    range_product = FenicsxMatrixOperator(range_product_mat, Vsub, Vsub)

    ns_vecs = build_nullspace(Vsub, gdim=submesh.geometry.dim)
    from dolfinx.fem.petsc import set_bc
    for vec in ns_vecs:
        set_bc(vec, bc_hom)
    range_space = FenicsxVectorSpace(Vsub)
    nullspace = range_space.make_array([ns_vecs[0], ns_vecs[2]])
    gram_schmidt(nullspace, product=range_product, copy=False)

    gamma_out = plane_at(0.0, "x")  # left plane
    tp = TransferProblem(
        problem,
        subproblem,
        gamma_out,
        dirichlet=dirichlet_bc,
        source_product={'product': 'l2', 'bcs': ()},
        range_product=range_product,
        kernel=nullspace,
    )

    # For now pass kernel=None
    # U should satisfy Dirichlet BCs
    # Remove kernel
    # U should satisfy Dirichlet BCs

    # generate boundary data
    D = tp.generate_random_boundary_data(2, distribution="normal")
    assert np.isclose(tp.source_gamma_out.dim, D.shape[-1])
    U = tp.solve(D)
    assert np.isclose(U.dim, tp.S_to_R.size)

    # check dirichlet dofs origin
    dofs = locate_dofs(x_dofs_vectorspace(Vsub), x_bottom_right, s_=np.s_[1::2])
    zero_dofs = U.dofs(dofs)
    assert np.allclose(zero_dofs, np.zeros_like(zero_dofs))

    # compute reference solutions
    u_arr = U.to_numpy()
    u_ex = np.zeros_like(u_arr)
    dof_indices = tp.bc_dofs_gamma_out
    for i, vector in enumerate(D):
        boundary_function = fem.Function(V)
        boundary_vector = boundary_function.vector
        boundary_vector.array[dof_indices] = vector

        bc_gamma_out = {
            "boundary": gamma_out,
            "value": boundary_function,
            "method": "geometrical",
        }
        u_exact = exact_solution(problem, [bc_gamma_out, dirichlet_bc], Vsub)
        u_ex[i, :] = u_exact

    # remove kernel of exact solution
    UEX = tp.range.from_numpy(u_ex)
    U_proj = orthogonal_part(
        UEX, tp.kernel, product=tp.range_product, orthonormal=True
    )
    zero_dofs = U_proj.dofs(dofs)
    assert np.allclose(zero_dofs, np.zeros_like(zero_dofs))

    # projection of orthogonal part onto basis should be zero
    Zero = project_array(U_proj, tp.kernel, product=tp.range_product, orthonormal=True)
    assert np.allclose(Zero.to_numpy(), np.zeros_like(Zero.to_numpy()))

    u_ex = U_proj.to_numpy()
    error = u_ex - u_arr
    norm = np.linalg.norm(error)
    assert norm < 1e-12


def test_remove_full_kernel():
    """Topology

    Ω = (0, 3) x (0, 3)
    Ω_in = (1, 1) x (2, 2)
    Γ_out = ∂Ω
    """

    target_subdomain = within_range([1, 1, 0], [2, 2, 0])

    n = 60
    gdim = 2
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(
            0.0,
            3.0,
            0.0,
            3.0,
            num_cells=(n, n),
            facet_tags = {"bottom": 1, "left": 2, "right": 3, "top": 4},
            recombine=True,
            out_file=tf.name,
        )
        square, _, facet_markers = gmshio.read_from_msh(
            tf.name, MPI.COMM_WORLD, gdim=gdim
        )
    degree = 1
    ve = element("P", square.basix_cell(), degree, shape=(2,))
    V = fem.functionspace(square, ve)

    domain = RectangularDomain(square, facet_tags=facet_markers)
    emod = fem.Constant(square, default_scalar_type(1.0))
    nu = fem.Constant(square, default_scalar_type(0.2))
    phases = LinearElasticMaterial(gdim, E=emod, NU=nu, plane_stress=True)
    problem = LinearElasticityProblem(domain, V, phases)
    # subdomain problem
    cells_submesh = mesh.locate_entities(domain.grid, 2, target_subdomain)
    submesh = mesh.create_submesh(domain.grid, 2, cells_submesh)[0]
    Vsub = fem.functionspace(submesh, ve)

    subdomain = RectangularSubdomain(1023, submesh)
    subproblem = LinElaSubProblem(subdomain, Vsub, phases)

    gamma_out = lambda x: np.full(x[0].shape, True, dtype=bool)  # noqa: E731

    facets_gamma_out = mesh.locate_entities_boundary(V.mesh, 1, gamma_out)

    # ### Range product & Nullspace
    inner_range_product = InnerProduct(Vsub, "h1")
    range_product_mat = inner_range_product.assemble_matrix()
    range_product = FenicsxMatrixOperator(range_product_mat, Vsub, Vsub)
    ns_vecs = build_nullspace(Vsub, gdim=submesh.geometry.dim)
    range_space = FenicsxVectorSpace(Vsub)
    nullspace = range_space.make_array(ns_vecs)
    gram_schmidt(nullspace, product=range_product, copy=False)

    tp = TransferProblem(
        problem,
        subproblem,
        gamma_out,
        dirichlet=[],
        source_product={"product": "mass", "bcs": ()},
        range_product=range_product,
        kernel=nullspace,
    )
    # generate boundary data
    D = tp.generate_random_boundary_data(10, distribution="normal")

    # solution with dummy material
    U_ = tp.solve(D)

    # solution after updating the material parameters
    new_material = ({"E": 210e3, "NU": 0.3},)
    tp.update_material(new_material)

    U = tp.solve(D)
    assert not np.isclose(U_[0].to_numpy().max(), U[0].to_numpy().max())
    u_arr = U.to_numpy()

    # compute reference solutions
    u_ex = np.zeros_like(u_arr)
    dof_indices = tp.bc_dofs_gamma_out
    for i, vector in enumerate(D):
        boundary_function = fem.Function(V)
        boundary_vector = boundary_function.vector
        boundary_vector.array[dof_indices] = vector

        bc_gamma_out = {
            "boundary": facets_gamma_out,
            "value": boundary_function,
            "method": "topological",
            "entity_dim": 1,
        }
        u_exact = exact_solution(problem, [bc_gamma_out], Vsub)
        u_ex[i, :] = u_exact

    # remove kernel of exact solution
    UEX = tp.range.from_numpy(u_ex)
    U_proj = orthogonal_part(
        UEX, tp.kernel, product=tp.range_product, orthonormal=True
    )

    u_ex = U_proj.to_numpy()
    error = u_ex - u_arr
    norm = np.linalg.norm(error)
    assert norm < 1e-12


if __name__ == "__main__":
    test_remove_trans_x_rot()
    test_remove_rot()
    test_remove_full_kernel()
