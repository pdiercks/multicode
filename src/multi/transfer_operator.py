from typing import Optional
import numpy as np
import numpy.typing as npt
from dolfinx import fem
from scipy.sparse import csr_array
from scipy.sparse.linalg import factorized
from multi.product import InnerProduct
from pymor.algorithms.basic import project_array
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorArray


def transfer_operator_subdomains_2d(A: npt.NDArray[np.float64], dirichlet_dofs: npt.NDArray[np.int32], target_dofs: npt.NDArray[np.int32]) -> NumpyMatrixOperator:
    """Builds transfer operator

    Args:
        A: The full operator over domain Ω.
        dirichlet_dofs: All DOFs of boundary Γ_out.
        target_dofs: All DOFs of the target subdomain.

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
    return NumpyMatrixOperator(transfer_operator)


def discretize_source_product(V: fem.FunctionSpaceBase, product: str, dirichlet_dofs: npt.NDArray[np.int32], bcs: Optional[list[fem.DirichletBC]] = []):
    """Discretizes source product for transfer operator.

    Args:
        V: FE space on oversampling domain.
        product: The inner product to use.
        dirichlet_dofs: DOFs on Gamma out.
        bcs: Optional homogeneous boundary conditions.

    """
    inner_product = InnerProduct(V, product, bcs=bcs)
    matrix = inner_product.assemble_matrix()
    A = csr_array(matrix.getValuesCSR()[::-1])
    source_matrix = A[dirichlet_dofs, :][:, dirichlet_dofs]
    return NumpyMatrixOperator(source_matrix)


class OrthogonallyProjectedOperator(Operator):
    """Represents the orthogonal projection of an operator.

    This operator is implemented as the concatenation of the application
    of the original operator and orthogonal projection of the image onto the subspace.

    """

    linear = False

    def __init__(self, operator: Operator, basis: VectorArray, product: Optional[Operator] = None, orthonormal: Optional[bool] = True):
        assert isinstance(operator, Operator)
        assert basis in operator.range
        assert (product is None
                or (isinstance(product, Operator)
                    and operator.range == product.source
                    and product.range == product.source))
        self.__auto_init(locals())
        self.range = operator.range
        self.source = operator.source
        self.linear = operator.linear

    def apply(self, U, mu=None):
        self.parameters.assert_compatible(mu)
        V = self.operator.apply(U, mu=mu)
        V_proj = project_array(V, self.basis, self.product, self.orthonormal)
        return V_proj

