import ufl
from typing import Optional, Union, Sequence
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import create_matrix, assemble_matrix


class InnerProduct(object):
    """Represents an inner product.

    Args:
        V: The FE space.
        product: The inner product.
        bcs: Optional Dirichlet BCs to apply.

    """

    def __init__(self, V: fem.FunctionSpaceBase, product: Union[str, ufl.Form], bcs: Optional[Sequence[fem.DirichletBC]] = None):
        if bcs is None:
            bcs = ()
        self.V = V
        self.bcs = bcs
        if isinstance(product, str):
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            names = ("euclidean", "mass", "l2", "h1-semi", "stiffness", "h1")
            if product in ("euclidean",):
                form = None
            elif product in ("mass", "l2"):
                form = ufl.inner(u, v) * ufl.dx
            elif product in ("h1-semi", "stiffness"):
                form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            elif product in ("h1",):
                form = ufl.inner(u, v) *ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            else:
                raise KeyError(
                    f"I don't know how to compute inner product with name '{product}'."
                    + f"You need to provide the UFL form yourself or choose one of {names}."
                )
        else:
            form = product
        self.form = form

    def assemble_matrix(self) -> Union[PETSc.Mat, None]:
        """returns the matrix (`PETSc.Mat`) representing the inner product or None
        in case euclidean product is used"""
        ufl_form = self.form
        if ufl_form is None:
            # such that product=None is equal to euclidean inner product
            # when using pymor code
            return None
        else:
            compiled_form = fem.form(ufl_form)
            A = create_matrix(compiled_form)
            A.zeroEntries()
            assemble_matrix(A, compiled_form, bcs=self.bcs)
            A.assemble()

            return A
