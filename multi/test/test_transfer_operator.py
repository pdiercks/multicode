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
    # # transfer operator
    # A = df.assemble(lhs_form)
    # dummy = df.Function(V)
    # bc_hom.zero_columns(A, dummy.vector(), 1.0)

    # all_dofs = np.arange(V.dim())
    # all_inner_dofs = np.setdiff1d(all_dofs, dofs_out)
    # Amat = df.as_backend_type(A).mat()
    # full_operator = csc_matrix(Amat.getValuesCSR()[::-1], shape=Amat.size)
    # operator = full_operator[:, all_inner_dofs][all_inner_dofs, :]

    # # factorization
    # matrix_shape = operator.shape
    # start = time.time()
    # operator = factorized(operator)
    # end = time.time()
    # print(f"factorization of {matrix_shape} matrix in {end-start}")

    # target_subdomain = TargetSubdomain()
    # target_subdomain_mesh = df.SubMesh(mesh, target_subdomain)
    # R = df.FunctionSpace(target_subdomain_mesh, V.ufl_element())
    # V_to_R = make_mapping(R, V)

    # # mapping from old to new dof numbers
    # newdofs = np.zeros((V.dim(),), dtype=int)
    # newdofs[all_inner_dofs] = np.arange(all_inner_dofs.size)
    # range_dofs = newdofs[V_to_R]

    # rhs_op = full_operator[:, dofs_out][all_inner_dofs, :]
    # start = time.time()
    # transfer_operator = -operator(rhs_op.todense())[range_dofs, :]
    # end = time.time()
    # print(f"applied operator to rhs in {end-start}")


if __name__ == "__main__":
    test()
