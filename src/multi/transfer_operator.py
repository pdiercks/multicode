from typing import Optional
import time
import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import factorized
from pymor.operators.numpy import NumpyMatrixOperator


def transfer_operator_subdomains_2d(A: npt.NDArray[np.float64], dirichlet_dofs: npt.NDArray[np.int32], target_dofs: npt.NDArray[np.int32], projection_matrix: Optional[npt.NDArray[np.float64]]=None) -> NumpyMatrixOperator:
    """Builds transfer operator

    Args:
        A: The full operator over domain Ω.
        dirichlet_dofs: All DOFs of boundary Γ_out.
        target_dofs: All DOFs of the target subdomain.
        projection_matrix: Matrix used for projection onto the nullspace of A.
        If not None, the kernel (i.e. rigid body modes in linear elasticity) will
        be removed from the solution.
    """
    # FIXME what format (csr, csc) should A have for efficient slicing?

    # dirichlet dofs associated with Γ_out
    num_dofs = A.shape[0]
    all_dofs = np.arange(num_dofs)
    all_inner_dofs = np.setdiff1d(all_dofs, dirichlet_dofs)

    full_operator = A.copy()
    operator = full_operator[:, all_inner_dofs][all_inner_dofs, :]

    # factorization
    matrix_shape = operator.shape
    start = time.time()
    operator = factorized(operator)
    end = time.time()
    print(f"factorization of {matrix_shape} matrix in {end-start}", flush=True)

    # mapping from old to new dof numbers
    newdofs = np.zeros((num_dofs,), dtype=int)
    newdofs[all_inner_dofs] = np.arange(all_inner_dofs.size)
    range_dofs = newdofs[target_dofs]

    rhs_op = full_operator[:, dirichlet_dofs][all_inner_dofs, :]
    start = time.time()
    transfer_operator = -operator(rhs_op.todense())[range_dofs, :] # type: ignore
    end = time.time()
    print(f"applied operator to rhs in {end-start}", flush=True)

    if projection_matrix is not None:
        # remove kernel
        assert transfer_operator.shape[0] == projection_matrix.shape[0]
        P = np.eye(transfer_operator.shape[0]) - projection_matrix
        transfer_operator = P.dot(transfer_operator)

    return NumpyMatrixOperator(transfer_operator)
