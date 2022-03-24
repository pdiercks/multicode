import dolfin as df
import numpy as np
from fenics_helpers.boundary import plane_at
from multi import Domain, LinearElasticityProblem


def test():
    mesh = df.UnitSquareMesh(8, 8)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    domain = Domain(mesh, 0)
    problem = LinearElasticityProblem(domain, V, 210e3, 0.3)

    left = plane_at(0.0)
    right = plane_at(1.0)
    problem.add_dirichlet_bc(left, df.Constant((0, 0)))
    problem.add_neumann_bc(right, df.Constant((1000, 0)))
    a = problem.get_form_lhs()
    L = problem.get_form_rhs()

    u = df.Function(V)
    df.solve(a == L, u, problem.dirichlet_bcs())

    assert np.sum(df.assemble(L)[:]) > 0.0
    assert df.assemble(a).array().shape == (V.dim(), V.dim())
    assert np.sum(np.abs(u.vector()[:])) > 0.0


if __name__ == "__main__":
    test()
