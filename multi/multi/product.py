import dolfin as df


class InnerProduct:
    """class to represent an inner product

    Parameters
    ----------
    V
        The finite element space.
    name : str, optional
        The name of the inner product. Supported values are
        euclidean, mass, l2, h1-semi, stiffness, h1.
    bcs : tuple, list, optional
        Boundary conditions to apply.
    form : ufl.form.Form, optional
        The weak form (UFL) of the inner product.

    """

    def __init__(self, V, name=None, bcs=(), form=None):
        if not isinstance(bcs, (list, tuple)):
            bcs = (bcs,)
        self.V = V
        self.name = name
        self.u = df.TrialFunction(V)
        self.v = df.TestFunction(V)
        self.bcs = bcs
        names = ("euclidean", "mass", "l2", "h1-semi", "stiffness", "h1")
        if name not in names:
            if form is None:
                raise KeyError(
                    f"I don't know how to compute inner product with name '{self.name}'."
                    + "You need to provide the form yourself or choose a supported value."
                )
        self.form = form

    def get_form(self):
        """returns the weak form of the inner product"""
        u = self.u
        v = self.v
        if self.name in ("euclidean",):
            form = None
        elif self.name in ("mass", "l2"):
            form = df.inner(u, v) * df.dx
        elif self.name in ("h1-semi", "stiffness"):
            form = df.inner(df.grad(u), df.grad(v)) * df.dx
        elif self.name in ("h1",):
            form = (df.inner(u, v) + df.inner(df.grad(u), df.grad(v))) * df.dx
        else:
            # form specified by user
            form = self.form
        return form

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
                # bc.apply does not preserve symmetry
                bc.zero_columns(matrix, vector, 1.0)
            return matrix
