from typing import Optional

import numpy as np
import numpy.typing as npt

from petsc4py import PETSc
import dolfinx as df

from multi.product import InnerProduct

from pymor.algorithms.basic import project_array
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray


def discretize_transfer_operator(
    A: PETSc.Mat, gamma_dofs: npt.NDArray, target_dofs: npt.NDArray
) -> PETSc.Mat:
    """Discretizes transfer operator.

    Args:
        A: The operator integrated over the full oversampling domain.
        gamma_dofs: DOFs associated with the boundary gamma out (Γ_out).
        target_dofs: DOFs associated with target subdomain Ω_in.

    Returns:
        T: The transfer operator.

    """

    num_dofs = A.size[0]
    all_dofs = np.arange(num_dofs)
    all_inner_dofs = np.setdiff1d(all_dofs, gamma_dofs)

    # mapping from old to new dof indices
    newdofs = np.zeros((num_dofs,), dtype=int)
    newdofs[all_inner_dofs] = np.arange(all_inner_dofs.size)
    range_dofs = newdofs[target_dofs]

    # extract submatrix inner dofs
    is_inner = PETSc.IS().createGeneral(all_inner_dofs.astype(np.int32))
    B = A.createSubMatrix(is_inner, is_inner)

    # extract submatrix as RHS
    is_gamma = PETSc.IS().createGeneral(gamma_dofs.astype(np.int32))
    R = A.createSubMatrix(is_inner, is_gamma)
    C = R.convert(PETSc.Mat.Type.DENSE)

    A.destroy()

    # create solver using MUMPS
    solver = PETSc.KSP().create()
    solver.setOperators(B)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.getPC().setFactorSolverType("mumps")
    solver.setFromOptions()

    # ### solve BY=C
    # create solution matrix Y
    Y = PETSc.Mat().createDense([B.size[0], C.size[1]])
    Y.setUp()
    solver.matSolve(C, Y)

    # does this help?
    solver.destroy()
    B.destroy()
    C.destroy()
    R.destroy()

    is_rows = PETSc.IS().createGeneral(range_dofs.astype(np.int32))
    is_cols = PETSc.IS().createGeneral(list(range(Y.size[1])))
    T = -Y.createSubMatrix(is_rows, is_cols)
    Y.destroy()

    return T


def discretize_source_product(
    V: df.fem.FunctionSpace,
    product: str,
    gamma_dofs: npt.NDArray[np.int32],
    bcs: Optional[list[df.fem.DirichletBC]] = [],
) -> PETSc.Mat:
    """Discretizes source product for transfer operator.

    Args:
        V: FE space on oversampling domain.
        product: The inner product to use.
        gamma_dofs: DOFs on Gamma out.
        bcs: Optional homogeneous boundary conditions.

    """
    inner_product = InnerProduct(V, product, bcs=bcs)
    matrix = inner_product.assemble_matrix()
    is_gamma = PETSc.IS().createGeneral(gamma_dofs.astype(np.int32))
    A = matrix.createSubMatrix(is_gamma, is_gamma)
    return A


class OrthogonallyProjectedOperator(Operator):
    """Represents the orthogonal projection of an operator.

    This operator is implemented as the concatenation of the application
    of the original operator and orthogonal projection of the image onto the subspace.

    """

    linear = False

    def __init__(
        self,
        operator: Operator,
        basis: VectorArray,
        product: Optional[Operator] = None,
        orthonormal: Optional[bool] = True,
    ):
        assert isinstance(operator, Operator)
        assert basis in operator.range
        assert product is None or (
            isinstance(product, Operator)
            and operator.range == product.source
            and product.range == product.source
        )
        self.__auto_init(locals())
        self.range = operator.range
        self.source = operator.source
        self.linear = operator.linear

    def apply(self, U, mu=None):
        self.parameters.assert_compatible(mu)
        V = self.operator.apply(U, mu=mu)
        V_proj = project_array(V, self.basis, self.product, self.orthonormal)
        return V_proj
