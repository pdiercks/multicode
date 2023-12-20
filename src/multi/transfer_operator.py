from typing import Optional
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array
from scipy.sparse.linalg import factorized
from multi.product import InnerProduct
from pymor.operators.numpy import NumpyMatrixOperator
from dolfinx import fem


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
    operator = factorized(operator)

    # mapping from old to new dof numbers
    newdofs = np.zeros((num_dofs,), dtype=int)
    newdofs[all_inner_dofs] = np.arange(all_inner_dofs.size)
    range_dofs = newdofs[target_dofs]

    rhs_op = full_operator[:, dirichlet_dofs][all_inner_dofs, :]
    transfer_operator = -operator(rhs_op.todense())[range_dofs, :] # type: ignore

    if projection_matrix is not None:
        # remove kernel
        assert transfer_operator.shape[0] == projection_matrix.shape[0]
        P = np.eye(transfer_operator.shape[0]) - projection_matrix
        transfer_operator = P.dot(transfer_operator)

    return NumpyMatrixOperator(transfer_operator)


def discretize_source_product(V: fem.FunctionSpaceBase, product: str, dirichlet_dofs: npt.NDArray[np.int32], bcs: Optional[list[fem.DirichletBC]] = []):
    """Discretizes source product for transfer operator.

    Args:
        V: FE space on oversampling domain.
        product: The inner product to use.
        dirichlet_dofs: DOFs on Gamma out.
        bcs: Optional homogeneous boundary conditions.

    """
    # FIXME double check if optional bcs are necessary | make a difference
    # If present, these should be far away from dirichlet_dofs and hence
    # corresponding entries in A are zero anyway?
    inner_product = InnerProduct(V, product, bcs=bcs)
    matrix = inner_product.assemble_matrix()
    ai, aj, av = matrix.getValuesCSR()
    A = csr_array((av, aj, ai))
    source_matrix = A[dirichlet_dofs, :][:, dirichlet_dofs]
    source_product = NumpyMatrixOperator(source_matrix, name=inner_product.name)
    return source_product
