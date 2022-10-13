"""test apply bcs"""

import numpy as np
from multi.bcs import apply_bcs


def boundary_expression():
    """Defines the function to be used for the boundary conditions"""
    return "1.0 + x[0] * x[0] + 2.0 * x[1] * x[1]"


def test():
    A = np.random.rand(10, 10)
    F = np.ones(10) * 12.2
    bc_dofs = [0, 2, 3, 4, 7, 9]
    bc_vals = [22.0, 4.1, 2, 0, 3.3, 10.01]
    inner_dofs = list(set(range(10)).difference(bc_dofs))

    rhs = np.zeros_like(F)
    for i in inner_dofs:
        rhs[i] = 12.2 - A[i, bc_dofs] @ np.array(bc_vals)

    apply_bcs(A, F, bc_dofs, bc_vals)

    assert np.allclose(F[inner_dofs], rhs[inner_dofs])
    assert np.allclose(F[bc_dofs], np.array(bc_vals))

    s = 0
    for dof in bc_dofs:
        s += np.sum(A[dof, :])
        s += np.sum(A[:, dof])
    assert s == len(bc_dofs) * 2

    U = np.linalg.solve(A, F)
    assert np.allclose(U[bc_dofs], bc_vals)

    # ### compare with dolfin
    # mesh = df.UnitSquareMesh(10, 10)
    # degree = 2
    # func_space = df.FunctionSpace(mesh, "CG", degree)
    # boundary_data = df.Expression(boundary_expression(), degree=2)

    # def boundary(_, on_boundary):
    #     return on_boundary

    # boundary_conditions = df.DirichletBC(func_space, boundary_data, boundary)
    # trial_function = df.TrialFunction(func_space)
    # test_function = df.TestFunction(func_space)
    # source = df.Constant(-6.0)
    # lhs = df.dot(df.grad(trial_function), df.grad(test_function)) * df.dx
    # rhs = source * test_function * df.dx

    # A = df.assemble(lhs)
    # A_array = A.array()
    # b = df.assemble(rhs)
    # b_array = b.get_local()

    # # reference
    # boundary_conditions.zero_columns(A, b, 1.0)
    # assert not np.allclose(A_array, A.array())
    # assert not np.allclose(b_array, b.get_local())

    # # apply_bcs
    # bc_dofs = list(boundary_conditions.get_boundary_values().keys())
    # bc_vals = list(boundary_conditions.get_boundary_values().values())
    # apply_bcs(A_array, b_array, bc_dofs, bc_vals)

    # assert np.allclose(A_array, A.array())
    # assert np.allclose(b_array, b.get_local())


if __name__ == "__main__":
    test()
