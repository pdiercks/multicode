"""test discretization of the transfer operator"""
import numpy as np
import dolfin as df
from multi.transfer_operator import transfer_operator_subdomains_2d


class DummyProblem:
    def __init__(self, V, lhs_form):
        self.V = V
        self.lhs_form = lhs_form

    def get_lhs(self):
        return self.lhs_form


class TargetSubdomain(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= 0.5 + 1e-3 and x[1] <= 0.5 + 1e-3


def test():
    mesh = df.UnitSquareMesh(6, 6)
    V = df.FunctionSpace(mesh, "CG", 1)

    def gamma_out(x, on_boundary):
        return df.near(x[1], 1.0) or df.near(x[0], 1.0) and on_boundary

    def sigma_d(x, on_boundary):
        return x[0] <= 0.5 and x[1] < 1e-2 and on_boundary

    bc_out = df.DirichletBC(V, df.Expression("x[0] * x[1]", degree=2), gamma_out)
    dofs_out = list(bc_out.get_boundary_values().keys())

    bc_hom = df.DirichletBC(V, df.Constant(0), sigma_d)

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    lhs_form = df.inner(df.grad(u), df.grad(v)) * df.dx
    rhs_form = df.Constant(0.0) * v * df.dx

    u_ref = df.Function(V)
    df.solve(lhs_form == rhs_form, u_ref, bcs=[bc_hom, bc_out])

    problem = DummyProblem(V, lhs_form)
    omega_in = TargetSubdomain()
    transfer_operator, _, _ = transfer_operator_subdomains_2d(
        problem, omega_in, gamma_out, bc_hom=bc_hom, product=None
    )

    # construct souce vector and solve
    g = df.Function(V)
    bc_out.apply(g.vector())
    source_vector = g.vector()[dofs_out]
    t_mat = transfer_operator.matrix
    u = t_mat.dot(source_vector)

    target_subdomain_mesh = df.SubMesh(mesh, omega_in)
    R = df.FunctionSpace(target_subdomain_mesh, V.ufl_element())

    u_ref_in = df.interpolate(u_ref, R)
    err = np.array(u).flatten() - u_ref_in.vector()[:]
    assert np.linalg.norm(err) < 1e-6


if __name__ == "__main__":
    test()
