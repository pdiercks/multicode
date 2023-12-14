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
from multi.boundary import plane_at, within_range
from multi.preprocessing import create_rectangle
from multi.projection import orthogonal_part


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
    u_in.interpolate(u_exact, nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
        u_in.function_space.mesh._cpp_object,
        u_in.function_space.element,
        u_exact.function_space.mesh._cpp_object))

    # clean up
    problem.clear_bcs()
    return u_in.vector.array


def test_dirichlet_hom():
    """Topology

    Ω = (0, 1) x (0, 1)
    Ω_in = (0.5, 1) x (0, 0.5)
    Γ_out = left boundary
    Σ_D_hom = right boundary
    """

    def target_subdomain(x):
        tol = 1e-4
        a = x[0] >= 0.5 - tol
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
            facets=True,
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
    phases = (LinearElasticMaterial(gdim, 210e3, 0.3, plane_stress=True),)
    problem = LinearElasticityProblem(domain, V, phases)
    # subdomain problem
    cells_submesh = mesh.locate_entities(domain.grid, 2, target_subdomain)
    submesh = mesh.create_submesh(domain.grid, 2, cells_submesh)[0]

    # submesh has same cell type, reuse ve
    Vsub = fem.functionspace(submesh, ve)

    subdomain = RectangularSubdomain(99, submesh)
    subproblem = LinElaSubProblem(subdomain, Vsub, phases)

    zero = fem.Constant(square, (default_scalar_type(0.0), default_scalar_type(0.0)))
    right = plane_at(1.0, "x")
    dirichlet_bc = {"boundary": right, "value": zero, "method": "geometrical"}
    gamma_out = plane_at(0.0, "x")  # left

    tp = TransferProblem(
        problem, subproblem, gamma_out, dirichlet=dirichlet_bc, 
    )
    # generate boundary data
    D = tp.generate_random_boundary_data(2, distribution='normal')
    assert np.isclose(tp.source_gamma_out.dim, D.shape[-1])
    U = tp.solve(D)
    assert np.isclose(U.dim, tp.S_to_R.size)
    u_arr = U.to_numpy()

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

    error = u_ex - u_arr
    norm = np.linalg.norm(error)
    assert norm < 1e-12


def test_remove_kernel():
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
            facets=True,
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
    phases = (LinearElasticMaterial(gdim, 210e3, 0.3, plane_stress=True),)
    problem = LinearElasticityProblem(domain, V, phases)
    # subdomain problem
    cells_submesh = mesh.locate_entities(domain.grid, 2, target_subdomain)
    submesh = mesh.create_submesh(domain.grid, 2, cells_submesh)[0]
    Vsub = fem.functionspace(submesh, ve)

    subdomain = RectangularSubdomain(1023, submesh)
    subproblem = LinElaSubProblem(subdomain, Vsub, phases)

    gamma_out = lambda x: np.full(x[0].shape, True, dtype=bool)  # noqa: E731

    facets_gamma_out = mesh.locate_entities_boundary(
        V.mesh, 1, gamma_out
    )

    tp = TransferProblem(
            problem, subproblem, gamma_out, dirichlet=[], source_product={"product": "mass"}, range_product={"product": "h1"}, remove_kernel=True
    )
    # generate boundary data
    D = tp.generate_random_boundary_data(10, distribution='normal')
    U = tp.solve(D)
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
    U_proj = orthogonal_part(tp.kernel, UEX, tp.range_l2_product, orth=True)

    u_ex = U_proj.to_numpy()
    error = u_ex - u_arr
    norm = np.linalg.norm(error)
    assert norm < 1e-12


if __name__ == "__main__":
    test_dirichlet_hom()
    test_remove_kernel()
