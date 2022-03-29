"""test discretization of the transfer operator"""
import numpy as np
import dolfin as df
from multi.transfer_operator import transfer_operator_subdomains_2d
from multi.product import InnerProduct


class DummyProblem:
    def __init__(self, V, lhs_form):
        self.V = V
        self.lhs_form = lhs_form

    def get_form_lhs(self):
        return self.lhs_form

    def discretize_product(self, name="energy", bcs=False):
        if bcs:
            raise NotImplementedError
        else:
            bcs = ()
        if name == "energy":
            product = InnerProduct(self.V, self.get_form_lhs(), bcs=bcs, name=name)
        else:
            product = InnerProduct(self.V, name, bcs=bcs)
        return product.assemble()


class TargetSubdomain(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= 0.5 + 1e-3 and x[1] <= 0.5 + 1e-3


def test():
    mesh = df.UnitSquareMesh(6, 6)
    V = df.FunctionSpace(mesh, "CG", 1)

    def gamma_out(x, on_boundary):
        """top or right"""
        return df.near(x[1], 1.0) or df.near(x[0], 1.0) and on_boundary

    def sigma_d(x, on_boundary):
        """bottom edge with x[0] <= 0.5"""
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
    submesh = df.SubMesh(mesh, omega_in)
    R = df.FunctionSpace(submesh, V.ufl_element())
    trial = df.TrialFunction(R)
    test = df.TestFunction(R)
    sub_lhs_form = df.inner(df.grad(trial), df.grad(test)) * df.dx
    sub_problem = DummyProblem(R, sub_lhs_form)
    transfer_operator, _, _ = transfer_operator_subdomains_2d(
        problem, sub_problem, gamma_out, bc_hom=bc_hom, product="energy"
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
