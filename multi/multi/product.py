import dolfin as df


class InnerProduct:
    """class to represent an inner product

    Parameters
    ----------
    V
        The finite element space.
    name : str
        The name of the product. E.g. 'mass' (or 'l2'),
        'stiffness' (or 'h1-semi') or 'h1' are known products.
    bcs : tuple, list, optional
        Boundary conditions to apply.
    form : None, optional
        The weak form of the product if name is not known. See name.

    """

    def __init__(self, V, name, bcs=(), form=None):
        if not isinstance(bcs, (list, tuple)):
            bcs = (bcs, )
        self.V = V
        self.name = name
        self.u = df.TrialFunction(V)
        self.v = df.TestFunction(V)
        self.bcs = bcs
        names = ('mass', 'l2', 'h1-semi', 'stiffness', 'h1')
        if name not in names:
            if not form:
                raise KeyError(f"I don't know how to compute product '{self.name}'."
                               + "You need ro provide the form yourself or choose another name.")
        self.form = form

    def get_form(self):
        """returns the weak form of the inner product"""
        u = self.u
        v = self.v
        if self.name in ('mass', 'l2'):
            form = df.inner(u, v) * df.dx
        elif self.name in ('h1-semi', 'stiffness'):
            form = df.inner(df.grad(u), df.grad(v)) * df.dx
        elif self.name in ('h1',):
            form = (df.inner(u, v) + df.inner(df.grad(u), df.grad(v))) * df.dx
        else:
            # e.g. energy product specified by user
            form = self.form
        return form

    def assemble(self):
        """returns the dolfin matrix representing the inner product"""
        product = df.assemble(self.get_form())
        for bc in self.bcs:
            bc.apply(product)
        return product
