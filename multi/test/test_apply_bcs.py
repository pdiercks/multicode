"""test apply bcs"""
import numpy as np
from multi.bcs import apply_bcs


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


if __name__ == "__main__":
    test()
