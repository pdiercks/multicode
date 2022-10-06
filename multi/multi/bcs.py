from dolfinx.mesh import exterior_facet_indices
from dolfinx.fem import (
    dirichletbc,
    locate_dofs_topological,
    locate_dofs_geometrical,
    function,
)
import ufl
import numpy as np

"""boundary conditions in fenicsx

Dirichlet
---------

V = FunctionSpace

Option (1): locate_dofs_topological
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = dolfinx.fem.dirichletbc(uD, boundary_dofs)

Option (2): locate_dofs_geometrical
def on_boundary(x):
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1)
boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)
bc = fem.dirichletbc(ScalarType(0), boundary_dofs, V)

V = VectorFunctionSpace

def clamped_boundary(x):
    return np.isclose(x[1], 0)

u_zero = np.array((0,)*mesh.geometry.dim, dtype=ScalarType)
bc = dirichletbc(u_zero, locate_dofs_geometrical(V, clamped_boundary), V)

Componet-wise
-------------

def right(x):
    return np.logical_and(np.isclose(x[0], L), x[1] < H)
boundary_facets = locate_entities_boundary(mesh, mesh.topology.dim-1, right)
boundary_dofs_x = locate_dofs_topological(V.sub(0), mesh.topology.dim-1, boundary_facets)
bcx = dirichletbc(ScalarType(0), boundary_dofs_x, V.sub(0))

"""

"""locate_dofs_topological(V, entity_dim, entities)
    V: iter(FunctionSpace)
    entity_dim: int
    entities: np.ndarray
"""

"""locate_dofs_geometrical(V, marker)
    V: iter(FunctionSpace)
    marker: callable
        A function that takes an array of points x with shape (gdim, num_points)
        and returns an array of booleans of length num_points evaluating to True
        for entities whose dof should be returned.
"""

"""fem.dirichletbc(value, dofs, V=None)
    value: Function, Constant or np.ndarray
    dofs: np.ndarray
    V: FunctionSpace; optional if value is Function
"""


def get_boundary_dofs(V):
    """get dof indices associated with the boundary of V.mesh"""
    domain = V.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    boundary_facets = exterior_facet_indices(domain.topology)
    dofs = locate_dofs_topological(V, fdim, boundary_facets)
    u = function.Function(V)
    u.x.set(0.0)
    bc = dirichletbc(u, dofs)
    return bc.dof_indices()[0]


# adapted version of MechanicsBCs by Thomas Titscher
class BoundaryConditions:
    """Handles dirichlet and neumann boundary conditions

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational domain of the problem.
    space : dolfinx.fem.FunctionSpace
        Finite element space defined on the domain.
    facet_tags : optional, dolfinx.mesh.meshtags
        A MeshTags object identifying the boundaries.

    """

    def __init__(self, domain, space, facet_tags=None):
        self.domain = domain
        self.V = space

        # list of dirichlet boundary conditions
        self._bcs = []

        # handle facets and measure for neumann bcs
        self._neumann_bcs = []
        self._facet_tags = facet_tags
        self._ds = ufl.Measure("ds", domain=domain, subdomain_data=self._facet_tags)
        self._v = ufl.TestFunction(space)

    def add_dirichlet_bc(
        self, value, boundary, sub=None, method="topological", entity_dim=None
    ):
        """add a Dirichlet BC

        Parameters
        ----------
        value : Function, Constant or np.ndarray
            The dirichlet data.
        boundary : callable or np.ndarray or int
            The part of the boundary whose dofs should be constrained.
            This can be a callable defining the boundary geometrically or
            an array of entity tags or an integer marking the boundary if
            `facet_tags` is not None.
        sub : optional, int
            If `sub` is not None the subspace `V.sub(sub)` will be constrained.
        method : optional, str
            A hint which method should be used to locate the dofs.
            Choice: 'topological' or 'geometrical'.
        entity_dim : optional, int
            The entity dimension in case `method=topological`.
        """

        assert method in ("topological", "geometrical")
        V = self.V.sub(sub) if sub is not None else self.V

        if method == "topological":
            if isinstance(boundary, int):
                try:
                    facets = self._facet_tags.find(boundary)
                except AttributeError as atterr:
                    raise atterr("There are no facet tags defined!")
            else:
                facets = boundary
            assert entity_dim is not None

            dofs = locate_dofs_topological(V, entity_dim, facets)
        else:
            dofs = locate_dofs_geometrical(V, boundary)

        # FIXME passing V if value is a Function raises TypeError
        # why is this case not covered by the 4th constructor of dirichletbc?
        if isinstance(value, function.Function):
            bc = dirichletbc(value, dofs)
        else:
            bc = dirichletbc(value, dofs, V)
        self._bcs.append(bc)

    def add_neumann_bc(self, marker, values):
        """adds a Neumann BC.

        Parameters
        ----------
        marker : int
        values : some ufl type
            The neumann data, e.g. traction vector.

        """
        self._neumann_bcs.append([values, marker])

    @property
    def has_neumann(self):
        return len(self._neumann_bcs) > 0

    @property
    def has_dirichlet(self):
        return len(self._bcs) > 0

    @property
    def bcs(self):
        """returns list of dirichlet bcs"""
        return self._bcs

    def clear(self, dirichlet=True, neumann=True):
        """clear list of boundary conditions"""
        if dirichlet:
            self._bcs.clear()
        if neumann:
            self._neumann_bcs.clear()

    @property
    def neumann_bcs(self):
        """returns ufl form of (sum of) neumann bcs"""
        r = 0
        for expression, marker in self._neumann_bcs:
            r += ufl.inner(expression, self._v) * self._ds(marker)
        return r


# def compute_multiscale_bcs(
#     problem, cell_index, edge_id, boundary_data, dofmap, chi, product=None, orth=False
# ):
#     """compute multiscale bcs from given boundary data

#     The dof values for the fine scale edge basis are computed
#     via projection of the boundary data onto the fine scale basis.
#     The coarse scale basis functions are assumed to be linear functions
#     with value 1 or 0 at the endpoints of the edge mesh.

#     Parameters
#     ----------
#     problem : multi.problems.LinearProblemBase
#         The linear problem.
#     cell_index : int
#         The cell index with respect to the coarse grid.
#     edge_id : int
#         Integer specifying the boundary edge.
#     boundary_data : dolfin.Expression or similar
#         The (total) displacement value on the boundary.
#         Will be projected onto the edge space.
#     dofmap : multi.dofmap.DofMap
#         The dofmap of the reduced order model.
#     chi : np.ndarray
#         The fine scale edge basis. ``chi.shape`` has to agree
#         with number of dofs for current coarse grid cell.
#     product : str or ufl.form.Form, optional
#         The inner product wrt which chi was orthonormalized.
#     orth : bool, optional
#         If True, assume orthonormal basis.


#     Returns
#     -------
#     bcs : dict
#         The dofs (dict.keys()) and values (dict.values()) of the
#         projected boundary conditions.

#     """
#     edge = problem.domain.edges[edge_id]
#     edge_space = problem.edge_spaces[edge_id]
#     source = FenicsVectorSpace(edge_space)

#     edge_coord = edge.coordinates()
#     edge_xmin = np.amin(edge_coord[:, 0])
#     edge_xmax = np.amax(edge_coord[:, 0])
#     edge_ymin = np.amin(edge_coord[:, 1])
#     edge_ymax = np.amax(edge_coord[:, 1])

#     bcs = {}  # dofs are keys, bc_values are values
#     coarse_vertices = np.array([[edge_xmin, edge_ymin], [edge_xmax, edge_ymax]])
#     coarse_values = np.hstack([boundary_data(point) for point in coarse_vertices])
#     coarse_dofs = dofmap.locate_dofs(coarse_vertices)

#     for dof, val in zip(coarse_dofs, coarse_values):
#         bcs.update({dof: val})

#     dofs_per_edge = dofmap.dofs_per_edge
#     if isinstance(dofs_per_edge, (int, np.integer)):
#         add_fine_bcs = dofs_per_edge > 0
#         edge_basis_length = dofs_per_edge
#     else:
#         assert dofs_per_edge.shape == (len(dofmap.cells), 4)
#         dofs_per_edge = dofs_per_edge[cell_index]
#         add_fine_bcs = np.sum(dofs_per_edge) > 0
#         edge_basis_length = dofs_per_edge[edge_id]

#     if add_fine_bcs:
#         # ### subtract coarse scale part from boundary data
#         g = df.interpolate(boundary_data, edge_space)
#         G = source.make_array([g.vector()])

#         component = 0 if edge_id in (0, 2) else 1
#         if edge_id in (0, 2):
#             # horizontal edge
#             component = 0
#             nodes = np.array([edge_xmin, edge_xmax])
#         else:
#             # vertical edge
#             component = 1
#             nodes = np.array([edge_ymin, edge_ymax])
#         line = NumpyLine(nodes)
#         phi_array = line.interpolate(edge_space, component)
#         phi = source.from_numpy(phi_array)
#         Gfine = G - phi.lincomb(coarse_values)

#         # ### build inner product for edge space
#         product_bc = df.DirichletBC(
#             edge_space, df.Function(edge_space), df.DomainBoundary()
#         )
#         inner_product = InnerProduct(edge_space, product, bcs=(product_bc,))
#         product = inner_product.assemble_operator()

#         edge_basis = source.from_numpy(chi)
#         edge_basis = edge_basis[:edge_basis_length]
#         if orth:
#             coeff = Gfine.inner(edge_basis, product=product)
#         else:
#             Gramian = edge_basis.gramian(product=product)
#             R = edge_basis.inner(Gfine, product=product)
#             coeff = np.linalg.solve(Gramian, R)
#         edge_mid_point = [
#             [
#                 (edge_xmax - edge_xmin) / 2 + edge_xmin,
#                 (edge_ymax - edge_ymin) / 2 + edge_ymin,
#             ]
#         ]
#         fine_dofs = dofmap.locate_dofs(edge_mid_point)

#         try:
#             coeff = coeff.reshape(fine_dofs.shape)
#         except ValueError as verr:
#             raise ValueError(
#                 "Number of modes per edge (dofmap) and the number of modes of the edge basis do not agree!"
#             ) from verr

#         for dof, val in zip(fine_dofs, coeff):
#             bcs.update({dof: val})

#     return bcs


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
