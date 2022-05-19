import dolfin as df
from pymor.bindings.fenics import FenicsMatrixOperator


class InnerProduct:
    """class to represent an inner product

    Parameters
    ----------
    V
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
            u = df.TrialFunction(V)
            v = df.TestFunction(V)
            names = ("euclidean", "mass", "l2", "h1-semi", "stiffness", "h1")
            if product in ("euclidean",):
                form = None
            elif product in ("mass", "l2"):
                form = df.inner(u, v) * df.dx
            elif product in ("h1-semi", "stiffness"):
                form = df.inner(df.grad(u), df.grad(v)) * df.dx
            elif product in ("h1",):
                form = (df.inner(u, v) + df.inner(df.grad(u), df.grad(v))) * df.dx
            else:
                raise KeyError(
                    f"I don't know how to compute inner product with name '{product}'."
                    + "You need to provide the form yourself or choose a supported value."
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

    def assemble(self):
        """returns the dolfin matrix representing the inner product or None
        in case euclidean product is used"""
        if self.get_form() is None:
            # such that product=None is equal to euclidean inner product
            # when using pymor code
            return None
        else:
            matrix = df.assemble(self.get_form())
            vector = df.Function(self.V).vector()
            for bc in self.bcs:
                # bc.apply does not preserce symmetry
                bc.zero_columns(matrix, vector, 1.0)
            return matrix

    def assemble_operator(self):
        """returns the dolfin matrix wrapped as FenicsMatrixOperator or None"""
        matrix = self.assemble()
        if matrix is not None:
            return FenicsMatrixOperator(matrix, self.V, self.V, name=self.name)
        else:
            return None
