# import yaml
from pathlib import Path
import dolfin as df
import numpy as np

from multi import Domain, LinearElasticityProblem
from multi.shapes import NumpyQuad
from multi.misc import get_solver
from fenics_helpers.boundary import plane_at
from multi.discr import discretize_block


def fenics_stuff(problem, mu, solver):
    domain = problem.domain
    V = problem.V

    xmin = domain.xmin
    xmax = domain.xmax
    ymin = domain.ymin
    ymax = domain.ymax

    def mid_point(a, b):
        assert b > a
        return a + (b - a) / 2

    points = [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
        [mid_point(xmin, xmax), ymin],
        [xmax, mid_point(ymin, ymax)],
        [mid_point(xmin, xmax), ymax],
        [xmin, mid_point(ymin, ymax)],
    ]

    quad = NumpyQuad(np.array(points))
    shapes = quad.interpolate(V)

    mu_arr = mu["mu"]
    mu_2 = mu_arr[12]
    mu_3 = mu_arr[13]
    mu_4 = mu_arr[4]

    fix = df.Function(V)
    fix.vector().set_local(np.zeros(V.dim()))

    pull = df.Function(V)
    pull.vector().set_local(mu_2 * shapes[12] + mu_3 * shapes[13] + mu_4 * shapes[4])

    problem.bc_handler.add_bc(plane_at(0, "y"), fix)
    problem.bc_handler.add_bc(plane_at(1, "y"), pull)

    u = problem.solve(solver_parameters=solver["solver_parameters"])
    return u


def test():
    BEAM = Path("/home/pdiercks/Repos/bam/2020_02_multiscale/beam")
    solver = get_solver(BEAM / "solver.yml")
    degree = 2
    mesh = df.UnitSquareMesh(8, 8)

    mu_arr = np.zeros(16)
    mu_arr[12] = 0.1
    mu_arr[13] = 0.1
    mu_arr[4] = 0.2
    mu = {"mu": mu_arr}

    mat = {"E": 210e3, "NU": 0.3}
    domain = Domain(mesh, 0, subdomains=False)
    V = df.VectorFunctionSpace(mesh, "CG", degree)
    problem = LinearElasticityProblem(
        domain, V, E=mat["E"], NU=mat["NU"], plane_stress=True
    )
    bcs = ({"boundary": plane_at(0, "y"), "value": df.Constant((0, 0))},)
    fom = discretize_block(
        problem,
        gamma=plane_at(1, "y"),
        serendipity=True,
        additional_bcs=bcs,
        solver=solver,
    )
    u = fenics_stuff(problem, mu, solver)
    U = fom.solve(mu)
    v = df.Function(fom.solution_space.V)
    v.vector().set_local(U.to_numpy().flatten())

    err = df.Function(fom.solution_space.V)
    err.vector().axpy(1.0, u.vector())
    err.vector().axpy(-1.0, v.vector())
    assert np.isclose(np.linalg.norm(err.vector()[:]), 0)


if __name__ == "__main__":
    test()
