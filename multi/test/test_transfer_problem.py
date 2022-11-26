"""test oversampling problem discretization for
inhomogeneous neumann boundary conditions"""

import tempfile
import numpy as np
import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py import PETSc
from multi.domain import RectangularDomain
from multi.problems import LinearElasticityProblem, TransferProblem
from multi.boundary import plane_at, within_range
from multi.preprocessing import create_rectangle_grid


def target_subdomain(x):
    tol = 1e-4
    a = x[0] >= 0.5 - tol
    b = x[1] <= 0.5 + tol
    return np.logical_and(a, b)


def exact_solution(problem, neumann_bc, dirichlet_bc, Vsub):
    """exact solution in full space"""
    problem.clear_bcs()

    if neumann_bc is not None:
        if isinstance(neumann_bc, list):
            for force in neumann_bc:
                problem.add_neumann_bc(**force)
        else:
            problem.add_neumann_bc(**neumann_bc)
    if dirichlet_bc is not None:
        if isinstance(dirichlet_bc, list):
            for bc in dirichlet_bc:
                problem.add_dirichlet_bc(**bc)
        else:
            problem.add_dirichlet_bc(**dirichlet_bc)

    # ### exact solution full space
    u_exact = problem.solve()

    u_in = dolfinx.fem.Function(Vsub)
    u_in.interpolate(u_exact)

    # clean up
    problem.clear_bcs()
    return u_in.vector.array


def test_dirichlet_neumann():
    """Topology

    Ω = (0, 1) x (0, 1)
    Ω_in = (0.5, 1) x (0, 0.5)
    Γ_out = left boundary
    Σ_N_inhom = bottom boundary
    Σ_N_hom = top boundary
    Σ_D_hom = right boundary
    """

    n = 20
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(
            0.0,
            1.0,
            0.0,
            1.0,
            num_cells=(n, n),
            facets=True,
            recombine=True,
            out_file=tf.name,
        )
        square, cell_markers, facet_markers = gmshio.read_from_msh(
            tf.name, MPI.COMM_WORLD, gdim=2
        )
    V = dolfinx.fem.VectorFunctionSpace(square, ("Lagrange", 1))

    domain = RectangularDomain(square, cell_markers=None, facet_markers=facet_markers)
    problem = LinearElasticityProblem(domain, V, E=210e3, NU=0.3, plane_stress=True)
    # subdomain problem
    cells_submesh = dolfinx.mesh.locate_entities(domain.grid, 2, target_subdomain)
    submesh = dolfinx.mesh.create_submesh(domain.grid, 2, cells_submesh)[0]
    Vsub = dolfinx.fem.FunctionSpace(submesh, problem.V.ufl_element())

    subdomain = RectangularDomain(submesh)
    subproblem = LinearElasticityProblem(subdomain, Vsub, E=210e3, NU=0.3, plane_stress=True)

    # marker=1 points to bottom
    traction = dolfinx.fem.Constant(
        square, (PETSc.ScalarType(0.0), PETSc.ScalarType(6e3))
    )
    zero = dolfinx.fem.Constant(square, (PETSc.ScalarType(0.0), PETSc.ScalarType(0.0)))
    neumann_bc = {"marker": 1, "value": traction}
    right = plane_at(1.0, "x")
    dirichlet_bc = {"boundary": right, "value": zero, "method": "geometrical"}
    gamma_out = plane_at(0.0, "x")  # left

    tp = TransferProblem(
        problem, subproblem, gamma_out, dirichlet=dirichlet_bc
    )
    tp.neumann = neumann_bc
    # generate boundary data
    randomState = np.random.RandomState(seed=6)
    D = tp.generate_random_boundary_data(2, random_state=randomState)
    U = tp.solve(D)
    u_arr = U.to_numpy()

    # compute reference solutions
    u_ex = np.zeros_like(u_arr)
    for i, vector in enumerate(D.to_numpy()):
        boundary_function = dolfinx.fem.Function(V)
        boundary_vector = boundary_function.vector
        boundary_vector.array[:] = vector

        bc_gamma_out = {
            "boundary": gamma_out,
            "value": boundary_function,
            "method": "geometrical",
        }
        u_exact = exact_solution(problem, neumann_bc, [bc_gamma_out, dirichlet_bc], Vsub)
        u_ex[i, :] = u_exact

    error = u_ex - u_arr
    norm = np.linalg.norm(error)
    print(norm)
    assert np.linalg.norm(error) < 1e-12


def test_neumann():
    """Topology

    Ω = (0, 1) x (0, 1)
    Ω_in = (0.5, 1) x (0, 0.5)
    Γ_out = union of left and top boundary
    Σ_N_inhom = bottom boundary
    Σ_N_hom = right boundary
    """

    n = 20
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(
            0.0,
            1.0,
            0.0,
            1.0,
            num_cells=(n, n),
            facets=True,
            recombine=True,
            out_file=tf.name,
        )
        square, cell_markers, facet_markers = gmshio.read_from_msh(
            tf.name, MPI.COMM_WORLD, gdim=2
        )
    V = dolfinx.fem.VectorFunctionSpace(square, ("Lagrange", 1))

    domain = RectangularDomain(square, cell_markers=None, facet_markers=facet_markers)
    problem = LinearElasticityProblem(domain, V, E=210e3, NU=0.3, plane_stress=True)
    # subdomain problem
    cells_submesh = dolfinx.mesh.locate_entities(domain.grid, 2, target_subdomain)
    submesh = dolfinx.mesh.create_submesh(domain.grid, 2, cells_submesh)[0]
    Vsub = dolfinx.fem.FunctionSpace(submesh, problem.V.ufl_element())

    subdomain = RectangularDomain(submesh)
    subproblem = LinearElasticityProblem(subdomain, Vsub, E=210e3, NU=0.3, plane_stress=True)

    traction = dolfinx.fem.Constant(
        square, (PETSc.ScalarType(0.0), PETSc.ScalarType(6e3))
    )
    neumann_bc = {"marker": 1, "value": traction}
    zero = dolfinx.fem.Constant(square, (PETSc.ScalarType(0.0), PETSc.ScalarType(0.0)))
    right = plane_at(1.0, "x")
    dirichlet_bc = {"boundary": right, "value": zero, "method": "geometrical"}

    def get_gamma_out(n):
        # mark the top and left boundary excluding points on bottom and right boundary
        Δx = Δy = 1.0 / (n + 1)  # must be smaller than cell size
        gamma_out = within_range([0.0, 0.0 + Δy, 0.0], [1.0 - Δx, 1.0, 0.0])
        return gamma_out

    gamma_out = get_gamma_out(n)

    facets_gamma_out = dolfinx.mesh.locate_entities_boundary(
        V.mesh, 1, gamma_out
    )

    tp = TransferProblem(
        problem, subproblem, gamma_out, dirichlet=dirichlet_bc
    )
    tp.neumann = neumann_bc
    # generate boundary data
    randomState = np.random.RandomState(seed=13)
    D = tp.generate_random_boundary_data(1, random_state=randomState)
    U = tp.solve(D)
    u_arr = U.to_numpy()

    # compute reference solutions
    u_ex = np.zeros_like(u_arr)
    for i, vector in enumerate(D.to_numpy()):
        boundary_function = dolfinx.fem.Function(V)
        boundary_vector = boundary_function.vector
        boundary_vector.array[:] = vector

        bc_gamma_out = {
            "boundary": facets_gamma_out,
            "value": boundary_function,
            "method": "topological",
            "entity_dim": 1,
        }
        u_exact = exact_solution(problem, neumann_bc, [bc_gamma_out, dirichlet_bc], Vsub)
        u_ex[i, :] = u_exact

    error = u_ex - u_arr
    norm = np.linalg.norm(error)
    print(norm)
    assert np.linalg.norm(error) < 1e-12


if __name__ == "__main__":
    test_dirichlet_neumann()
    test_neumann()
