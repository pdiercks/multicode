import dolfin as df
import numpy as np


# adapted version of MechanicsBCs by Thomas Titscher
class BoundaryConditions:
    """Handles dirichlet and neumann boundary conditions

    Parameters
    ----------
    domain
        Computational domain of the problem. An instance of |multi.Domain|.
    space
        Finite element space defined on the domain.

    """

    def __init__(self, domain, space):
        self._domain = domain
        self._V = space

        self._bc_expressions = []

        # list of dirichlet boundary conditions
        self._bcs = []

        # for neumann boundary conditions
        mesh = space.mesh()
        self._v = df.TestFunction(space)
        self._neumann_bcs = []
        self._boundary_markers = df.MeshFunction(
            "size_t", mesh, mesh.topology().dim() - 1, 0
        )
        self._ds_marker = 0
        self._ds = df.Measure("ds", domain=mesh, subdomain_data=self._boundary_markers)
        # number of components of the (vector) field
        self._ncomp = self._V.ufl_element().value_size()

    def _value_to_expression(self, value, degree):
        if isinstance(value, (list, tuple)):
            assert all([isinstance(v, str) for v in value])
            # transform list of strings to expression
            return df.Expression(value, t=0.0, degree=degree)

        if isinstance(value, str):
            return df.Expression(value, t=0.0, degree=degree)

        # in all other cases, we try to use `value` directly as expression
        return value

    def set_zero(self, boundary, sub=None, method="topological"):
        """
        Adds the dirichlet BC u[sub] = 0. If `sub` is None, it will constrain
        the whole vector [ux uy uz] = [0 0 0]
        """
        if self._ncomp == 1:
            self.add_dirichlet(boundary, "0", 0, None, method)
        else:
            self.add_dirichlet(boundary, ["0"] * self._ncomp, 0, sub, method)

    def add_dirichlet(self, boundary, value, degree=0, sub=None, method="topological"):
        """Adds a dirchlet BC. `value` may be a string, a collection of strings or
        something that behaves like a dolfin.Expression. (e.g. dolfin.Constant).
        If this `value` contains a `t`, it will be treated as a time dependent
        expression and modified to tx when `update(tx)` is called.

        Parameters
        ----------
        boundary
            The id of the boundary or the boundary itself to apply
            the dirichlet data to.
        value
            The dirichlet data.

        """
        expression = self._value_to_expression(value, degree)

        if hasattr(expression, "t"):
            self._bc_expressions.append(expression)

        space = self._V if sub is None else self._V.sub(sub)
        if isinstance(boundary, int):
            bc = df.DirichletBC(space, expression, self._domain.boundaries, boundary)
        else:
            bc = df.DirichletBC(space, expression, boundary, method)
        self._bcs.append(bc)

    def add_neumann(self, boundary, value, degree=0):
        """Adds a neumann BC. `value` may be a string, a collection of strings or
        something that behaves like a dolfin.Expression. (e.g. dolfin.Constant).
        If this `value` contains a `t`, it will be treated as a time dependent
        expression and modified to tx when `update(tx)` is called.

        Parameters
        ----------
        value
            The neumann data, traction vector.
        boundary
            The boundary to apply the neumann data to.

        """
        expression = self._value_to_expression(value, degree)

        if hasattr(expression, "t"):
            self._bc_expressions.append(expression)

        self._ds_marker += 1
        boundary.mark(self._boundary_markers, self._ds_marker)
        self._neumann_bcs.append([expression, self._ds_marker])

    def has_neumann(self):
        return len(self._neumann_bcs) > 0

    def update(self, t):
        for bc_expression in self._bc_expressions:
            bc_expression.t = t

    def bcs(self):
        """returns list of dirichlet bcs"""
        return self._bcs

    def clear(self, dirichlet=True, neumann=True):
        """clear list of boundary conditions"""
        if dirichlet:
            self._bcs.clear()
        if neumann:
            self._neumann_bcs.clear()

    def neumann_bcs(self):
        """returns ufl form of (sum of) neumann bcs"""
        r = 0
        for expression, marker in self._neumann_bcs:
            r += df.dot(expression, self._v) * self._ds(marker)
        return r


def apply_bcs(lhs, rhs, bc_indices, bc_values):
    """
    Applies dirichlet bcs (in-place) using the algorithm described here
    http://www.math.colostate.edu/~bangerth/videos/676/slides.21.65.pdf

    Parameters
    ----------
    lhs
        The left hand side of the system.
    rhs
        The right hand side of the system.
    bc_indices
        DOF indices where bcs should be applied.
    bc_values
        The boundary data.

    Returns
    -------
    None
    """
    assert isinstance(lhs, np.ndarray)
    assert isinstance(rhs, np.ndarray)
    assert isinstance(bc_indices, (list, np.ndarray))
    if isinstance(bc_indices, list):
        bc_indices = np.array(bc_indices)
    assert isinstance(bc_values, (list, np.ndarray))
    if isinstance(bc_values, list):
        bc_values = np.array(bc_values)

    rhs.shape = (rhs.size,)
    values = np.zeros(rhs.size)
    values[bc_indices] = bc_values
    # substract bc values from right hand side
    rhs -= np.dot(lhs[:, bc_indices], values[bc_indices])
    # set columns to zero
    lhs[:, bc_indices] = np.zeros((rhs.size, bc_indices.size))
    # set rows to zero
    lhs[bc_indices, :] = np.zeros((bc_indices.size, rhs.size))
    # set diagonal entries to 1.
    lhs[bc_indices, bc_indices] = np.ones(bc_indices.size)
    # set bc_values on right hand side
    rhs[bc_indices] = values[bc_indices]
