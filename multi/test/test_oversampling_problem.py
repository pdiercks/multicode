"""test oversampling problem discretization for
inhomogeneous neumann boundary conditions"""

import numpy as np
import dolfin as df
from multi import Domain, LinearElasticityProblem
from multi.oversampling import OversamplingProblem
from fenics_helpers.boundary import plane_at


class GammaOut(df.SubDomain):
    def inside(self, x, on_boundary):
        top_boundary = plane_at(1.0, "y")
        left_boundary = plane_at(0.0, "x")
        return top_boundary.inside(x, on_boundary) or left_boundary.inside(
            x, on_boundary
        )


class TargetSubDomain(df.SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-4
        return x[0] >= 0.5 - tol and x[1] <= 0.5 + tol


def exact_solution(problem, neumann_bc, dirichlet_bc):
    """exact solution in full space"""
    if isinstance(neumann_bc, list):
        for force in neumann_bc:
            problem.bc_handler.add_force(**force)
    else:
        problem.bc_handler.add_force(**neumann_bc)
    if isinstance(dirichlet_bc, list):
        for bc in dirichlet_bc:
            problem.bc_handler.add_bc(**bc)
    else:
        problem.bc_handler.add_bc(**dirichlet_bc)

    # ### exact solution full space
    u_exact = problem.solve(solver_parameters={"linear_solver": "mumps"})

    # clean up
    problem.bc_handler.remove_bcs()
    problem.bc_handler.remove_forces()
    return u_exact


def test_dirichlet_neumann():
    """Topology

    Ω = (0, 1) x (0, 1)
    Ω_in = (0.5, 1) x (0, 0.5)
    Γ_out = left boundary
    Σ_N_inhom = bottom boundary
    Σ_N_hom = top boundary
    Σ_D_hom = right boundary
    """

    mesh = df.UnitSquareMesh(20, 20)
    V = df.VectorFunctionSpace(mesh, "CG", 1)

    sigma_inhom = plane_at(0.0, "y")
    sigma_d = plane_at(1.0, "x")

    domain = Domain(mesh)
    problem = LinearElasticityProblem(domain, V, E=210e3, NU=0.3, plane_stress=True)
    neumann_bc = {"boundary": sigma_inhom, "value": df.Constant((0.0, 6e3))}
    dirichlet_bc = {"boundary": sigma_d, "value": df.Constant(0.0), "sub": 0}
    # ### data on Γ_out for exact solution
    gamma_out = plane_at(0.0, "x")

    # NOTE pymor solver options have different format
    os_problem = OversamplingProblem(
        problem,
        gamma_out,
        dirichlet=dirichlet_bc,
        neumann=neumann_bc,
        solver_options={"inverse": {"solver": "mumps"}},
    )
    # generate boundary data
    randomState = np.random.RandomState(seed=6)
    D = os_problem.generate_random_boundary_data(2, random_state=randomState)
    U = os_problem.solve(D)
    u_arr = U.to_numpy()

    # compute reference solutions
    u_ex = np.zeros_like(u_arr)
    for i, vector in enumerate(D.to_numpy()):
        boundary_function = df.Function(V)
        boundary_vector = boundary_function.vector()
        boundary_vector.set_local(vector)

        bc_gamma_out = {"boundary": gamma_out, "value": boundary_function}
        u_exact = exact_solution(problem, neumann_bc, [bc_gamma_out, dirichlet_bc])
        u_ex[i, :] = u_exact.vector()[:]

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

    class GammaOut(df.SubDomain):
        def inside(self, x, on_boundary):
            top_boundary = plane_at(1.0, "y")
            left_boundary = plane_at(0.0, "x")
            return top_boundary.inside(x, on_boundary) or left_boundary.inside(
                x, on_boundary
            )

    mesh = df.UnitSquareMesh(20, 20)
    V = df.VectorFunctionSpace(mesh, "CG", 1)

    sigma_inhom = plane_at(0.0, "y")
    # sigma_hom = plane_at(1.0, "x")

    domain = Domain(mesh)
    problem = LinearElasticityProblem(domain, V, E=210e3, NU=0.3, plane_stress=True)
    neumann_bc = {"boundary": sigma_inhom, "value": df.Constant((0.0, 6e3))}
    # ### data on Γ_out for exact solution
    gamma_out = GammaOut()

    # NOTE pymor solver options have different format
    os_problem = OversamplingProblem(
        problem,
        gamma_out,
        dirichlet=None,
        neumann=neumann_bc,
        solver_options={"inverse": {"solver": "mumps"}},
    )
    # generate boundary data
    randomState = np.random.RandomState(seed=6)
    D = os_problem.generate_random_boundary_data(10, random_state=randomState)
    U = os_problem.solve(D)
    u_arr = U.to_numpy()

    # compute reference solutions
    u_ex = np.zeros_like(u_arr)
    for i, vector in enumerate(D.to_numpy()):
        boundary_function = df.Function(V)
        boundary_vector = boundary_function.vector()
        boundary_vector.set_local(vector)

        bc_gamma_out = {"boundary": gamma_out, "value": boundary_function}
        u_exact = exact_solution(problem, neumann_bc, bc_gamma_out)
        u_ex[i, :] = u_exact.vector()[:]

    error = u_ex - u_arr
    norm = np.linalg.norm(error)
    print(norm)
    assert np.linalg.norm(error) < 1e-12


if __name__ == "__main__":
    test_dirichlet_neumann()
    test_neumann()
