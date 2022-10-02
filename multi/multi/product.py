import ufl
from dolfinx import fem


class InnerProduct(object):
    """class to represent an inner product

    Parameters
    ----------
    V : dolfinx.fem.function.FunctionSpace
        The finite element space.
    product : str or ufl.form.Form, optional
        The inner product given as string or weak form expressed in UFL.
        Supported string values are euclidean, mass, l2, h1-semi, stiffness, h1.
    bcs : tuple, list, optional
        Boundary conditions to apply.
    name : str, optional
        The name that will be given to the FenicsMatrixOperator when
        self.assemble_operator is called.

    """

    def __init__(self, V, product=None, bcs=(), name=None):
        if not isinstance(bcs, (list, tuple)):
            bcs = (bcs,)
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
                form = (ufl.inner(u, v) + ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
            else:
                raise KeyError(
                    f"I don't know how to compute inner product with name '{product}'."
                    + f"You need to provide the UFL form yourself or choose one of {names}."
                )
            self.name = name or product
        else:
            # product should be either an ufl.form.Form or None
            form = product
            self.name = name
        self.form = form

    def get_form(self):
        """returns the weak form of the inner product"""
        return self.form

    def assemble_matrix(self):
        """returns the matrix (`PETSc.Mat`) representing the inner product or None
        in case euclidean product is used"""
        ufl_form = self.get_form()
        if ufl_form is None:
            # such that product=None is equal to euclidean inner product
            # when using pymor code
            return None
        else:
            # TODO pass `form_compiler_params` and `jit_params`?
            # create dolfinx.fem.form.Form from ufl form
            compiled_form = fem.form(ufl_form)
            A = fem.petsc.create_matrix(compiled_form)
            fem.petsc.assemble_matrix(A, compiled_form, bcs=self.bcs)
            A.assemble()

            return A
