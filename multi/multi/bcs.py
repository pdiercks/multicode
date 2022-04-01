import dolfin as df
import numpy as np
from multi.product import InnerProduct
from multi.shapes import NumpyLine
from pymor.bindings.fenics import FenicsVectorSpace


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


def compute_multiscale_bcs(
    problem, edge_id, boundary_data, dofmap, chi, product=None, orth=False
):
    """compute multiscale bcs from given boundary data

    The dof values for the fine scale edge basis are computed
    via projection of the boundary data onto the fine scale basis.
    The coarse scale basis functions are assumed to be linear functions
    with value 1 or 0 at the endpoints of the edge mesh.

    Parameters
    ----------
    problem : multi.problems.LinearProblemBase
        The linear problem.
    edge_id : int
        Integer specifying the boundary edge.
    boundary_data : dolfin.Expression or similar
        The (total) displacement value on the boundary.
        Will be projected onto the edge space.
    dofmap : multi.dofmap.DofMap
        The dofmap of the reduced order model.
    chi : np.ndarray
        The fine scale edge basis. ``chi.shape`` has to agree
        with number of dofs for current coarse grid cell.
    product : str or ufl.form.Form, optional
        The inner product wrt which chi was orthonormalized.
    orth : bool, optional
        If True, assume orthonormal basis.


    Returns
    -------
    bcs : dict
        The dofs (dict.keys()) and values (dict.values()) of the
        projected boundary conditions.

    """
    edge = problem.domain.edges[edge_id]
    edge_space = problem.edge_spaces[edge_id]
    source = FenicsVectorSpace(edge_space)

    edge_coord = edge.coordinates()
    edge_xmin = np.amin(edge_coord[:, 0])
    edge_xmax = np.amax(edge_coord[:, 0])
    edge_ymin = np.amin(edge_coord[:, 1])
    edge_ymax = np.amax(edge_coord[:, 1])

    bcs = {}  # dofs are keys, bc_values are values
    coarse_vertices = np.array([[edge_xmin, edge_ymin], [edge_xmax, edge_ymax]])
    coarse_values = np.hstack([boundary_data(point) for point in coarse_vertices])
    coarse_dofs = dofmap.locate_dofs(coarse_vertices)

    for dof, val in zip(coarse_dofs, coarse_values):
        bcs.update({dof: val})

    dofs_per_edge = dofmap.dofs_per_edge
    if isinstance(dofs_per_edge, (int, np.integer)):
        add_fine_bcs = dofs_per_edge > 0
    else:
        add_fine_bcs = np.sum(dofs_per_edge) > 0

    if add_fine_bcs:
        # ### subtract coarse scale part from boundary data
        g = df.interpolate(boundary_data, edge_space)
        G = source.make_array([g.vector()])

        component = 0 if edge_id in (0, 2) else 1
        if edge_id in (0, 2):
            # horizontal edge
            component = 0
            nodes = np.array([edge_xmin, edge_xmax])
        else:
            # vertical edge
            component = 1
            nodes = np.array([edge_ymin, edge_ymax])
        line = NumpyLine(nodes)
        phi_array = line.interpolate(edge_space, component)
        phi = source.from_numpy(phi_array)
        Gfine = G - phi.lincomb(coarse_values)

        # ### build inner product for edge space
        product_bc = df.DirichletBC(
            edge_space, df.Function(edge_space), df.DomainBoundary()
        )
        inner_product = InnerProduct(edge_space, product, bcs=(product_bc,))
        product = inner_product.assemble_operator()

        edge_basis = source.from_numpy(chi)
        if orth:
            coeff = Gfine.inner(edge_basis, product=product)
        else:
            Gramian = edge_basis.gramian(product=product)
            R = edge_basis.inner(Gfine, product=product)
            coeff = np.linalg.solve(Gramian, R)
        edge_mid_point = [
            [
                (edge_xmax - edge_xmin) / 2 + edge_xmin,
                (edge_ymax - edge_ymin) / 2 + edge_ymin,
            ]
        ]
        fine_dofs = dofmap.locate_dofs(edge_mid_point)

        for dof, val in zip(fine_dofs, coeff.flatten().tolist()):
            bcs.update({dof: val})

    return bcs


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
