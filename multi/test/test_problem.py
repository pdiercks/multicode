import dolfin as df
import numpy as np
from fenics_helpers.boundary import plane_at
from multi import Domain, LinearElasticityProblem


def test():
    mesh = df.UnitSquareMesh(8, 8)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    domain = Domain(mesh, 0)
    problem = LinearElasticityProblem(domain, V, 210e3, 0.3)

    a = problem.get_lhs()
    L = problem.get_rhs()

    left = plane_at(0.0)
    right = plane_at(1.0)

    problem.bc_handler.add_bc(left, df.Constant((0, 0)))
    problem.bc_handler.add_force(right, df.Constant((1000, 0)))
    u = problem.solve()

    assert np.sum(df.assemble(L)[:]) == 0
    assert df.assemble(a).array().shape == (V.dim(), V.dim())
    assert np.sum(np.abs(u.vector()[:])) > 0.0


if __name__ == "__main__":
    test()
