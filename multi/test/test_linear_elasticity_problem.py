import dolfin as df
import numpy as np
from fenics_helpers.boundary import plane_at
from multi import Domain, RectangularDomain, LinearElasticityProblem


def test():
    domain = Domain(df.UnitSquareMesh(8, 8), _id=1)
    V = df.VectorFunctionSpace(domain.mesh, "CG", 1)
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


def test_with_edges():
    domain = RectangularDomain(
        "data/rvedomain.xdmf", _id=1, subdomains=False, edges=True
    )
    V = df.VectorFunctionSpace(domain.mesh, "CG", 1)
    problem = LinearElasticityProblem(domain, V, 210e3, 0.3)

    x_dofs = problem.V.tabulate_dof_coordinates()
    bottom = x_dofs[problem.V_to_L[0]]
    assert np.allclose(bottom[:, 1], np.zeros_like(bottom[:, 1]))
    left = x_dofs[problem.V_to_L[3]]
    assert np.allclose(left[:, 0], np.zeros_like(left[:, 0]))

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
    test_with_edges()
