"""test oversampling problem discretization for
inhomogeneous neumann boundary conditions"""

import numpy as np
import dolfin as df
from multi import RectangularDomain, LinearElasticityProblem
from multi.problems import OversamplingProblem
from multi.transfer_operator import transfer_operator_subdomains_2d
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
    u_exact = problem.solve(solver_options={"solver": "mumps"})

    # ### exact solution in range space
    omega_in = TargetSubDomain()
    submesh = df.SubMesh(problem.domain.mesh, omega_in)
    Vsub = df.FunctionSpace(submesh, problem.V.ufl_element())
    u_ex = df.interpolate(u_exact, Vsub)

    # clean up
    problem.clear_bcs()
    return u_ex


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

    domain = RectangularDomain(mesh)
    problem = LinearElasticityProblem(domain, V, E=210e3, NU=0.3, plane_stress=True)
    # subdomain problem
    omega_in = TargetSubDomain()
    submesh = df.SubMesh(mesh, omega_in)
    subdomain = RectangularDomain(submesh)
    Vsub = df.FunctionSpace(submesh, V.ufl_element())
    subproblem = LinearElasticityProblem(
        subdomain, Vsub, E=210e3, NU=0.3, plane_stress=True
    )
    neumann_bc = {"boundary": sigma_inhom, "value": df.Constant((0.0, 6e3))}
    dirichlet_bc = {"boundary": sigma_d, "value": df.Constant(0.0), "sub": 0}
    gamma_out = plane_at(0.0, "x")

    # NOTE pymor solver options have different format
    os_problem = OversamplingProblem(
        problem,
        subproblem,
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

    domain = RectangularDomain(mesh)
    problem = LinearElasticityProblem(domain, V, E=210e3, NU=0.3, plane_stress=True)
    # subdomain problem
    omega_in = TargetSubDomain()
    submesh = df.SubMesh(mesh, omega_in)
    subdomain = RectangularDomain(submesh)
    Vsub = df.FunctionSpace(submesh, V.ufl_element())
    subproblem = LinearElasticityProblem(
        subdomain, Vsub, E=210e3, NU=0.3, plane_stress=True
    )
    sigma_inhom = plane_at(0.0, "y")
    neumann_bc = {"boundary": sigma_inhom, "value": df.Constant((0.0, 6e3))}
    gamma_out = GammaOut()

    # NOTE pymor solver options have different format
    os_problem = OversamplingProblem(
        problem,
        subproblem,
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


def test_compare():
    """
    compare against transfer operator implementation
    and exact solution

    Topology

    Ω = (0, 1) x (0, 1)
    Ω_in = (0.5, 1) x (0, 0.5)
    Γ_out = union of left and top boundary
    Σ_N_hom = bottom and right boundary
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

    domain = RectangularDomain(mesh)
    problem = LinearElasticityProblem(domain, V, E=210e3, NU=0.3, plane_stress=True)
    # subdomain problem
    omega_in = TargetSubDomain()
    submesh = df.SubMesh(mesh, omega_in)
    subdomain = RectangularDomain(submesh)
    Vsub = df.FunctionSpace(submesh, V.ufl_element())
    subproblem = LinearElasticityProblem(
        subdomain, Vsub, E=210e3, NU=0.3, plane_stress=True
    )
    gamma_out = GammaOut()

    # NOTE pymor solver options have different format
    os_problem = OversamplingProblem(
        problem,
        subproblem,
        gamma_out,
        dirichlet=None,
        neumann=None,
        solver_options={"inverse": {"solver": "mumps"}},
    )
    # generate boundary data
    randomState = np.random.RandomState(seed=6)
    D = os_problem.generate_random_boundary_data(1, random_state=randomState)
    U = os_problem.solve(D)
    u_arr = U.to_numpy()

    # compute reference solutions
    u_ex = np.zeros_like(u_arr)
    for i, vector in enumerate(D.to_numpy()):
        boundary_function = df.Function(V)
        boundary_vector = boundary_function.vector()
        boundary_vector.set_local(vector)

        bc_gamma_out = {"boundary": gamma_out, "value": boundary_function}
        u_exact = exact_solution(problem, None, bc_gamma_out)
        u_ex[i, :] = u_exact.vector()[:]

    error = u_ex - u_arr
    norm = np.linalg.norm(error)
    print(f"Norm OSP against exact: {norm}")
    assert np.linalg.norm(error) < 1e-12

    # ### transfer operator
    top, _, _ = transfer_operator_subdomains_2d(
        problem, subproblem, gamma_out, bc_hom=None, product="h1"
    )
    bc_Γ = df.DirichletBC(problem.V, boundary_function, gamma_out)
    dofs = list(bc_Γ.get_boundary_values().keys())
    source_vector = D.dofs(dofs)
    Tv = top.apply(top.source.from_numpy(source_vector))
    error = u_ex - Tv.to_numpy()
    norm = np.linalg.norm(error)
    print(f"Norm TOP against exact: {norm}")
    assert np.linalg.norm(error) < 1e-12


if __name__ == "__main__":
    test_compare()
    test_dirichlet_neumann()
    test_neumann()
