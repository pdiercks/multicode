import dolfinx
import ufl
import numpy as np
from petsc4py import PETSc
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from multi.interpolation import interpolate
from multi.shapes import NumpyLine
from multi.product import InnerProduct

"""Dirichlet bcs online

Homogeneous
-----------
1. g=0
- set zero for coarse scale dofs
- do nothing for fine scale dofs (there should not exist any modes)
2. g_sub=0
- set zero for coarse scale dofs
- do nothing for fine scale dofs (there should exist a number of modes with zero values for sub and non-zero (free) for other; unless user specified incorrect pod config)

Inhomogeneous
-------------
3. g=const.=α
- set α for coarse scale dofs
- do nothing for fine scale dofs (there should not exist any modes)
4. g_sub=const.=α
- set α for coarse scale dofs
- do nothing for fine scale dofs (there should exist a number of modes with zero values for sub and non-zero (free) for other; unless user specified incorrect pod config)
5. g=non-zero
- set value for coarse scale dofs
- set 1. for fine scale dof (there should exist 1 mode on each boundary edge)
6. g_sub=non-zero
- set value for coarse scale dofs
- set 1. for fine scale dof (there should exist 1 mode on each boundary edge)


"""


def compute_dirichlet_online(dofmap, dirichlet):
    grid = dofmap.grid
    bcs = {}
    for bc in dirichlet["inhomogeneous"]:
        locator = bc["boundary"]
        uD = bc["value"]
        sub = bc.get("sub")

        # locate entities
        vertices = grid.locate_entities_boundary(0, locator)
        edges = grid.locate_entities_boundary(1, locator)
        # entity coordinates
        x_verts = grid.get_entity_coordinates(0, vertices)

        if sub is not None:
            assert sub in (0, 1)
            # coarse scale dofs
            coarse_values = interpolate(uD, x_verts)
            for values, ent in zip(coarse_values, vertices):
                dofs = dofmap.entity_dofs(0, ent)
                bcs.update({dofs[sub]: values[sub]})

            # fine scale dofs
            # g_sub=const.  --> mode_sub=0 but non-zero otherwise
            # g_sub=anything --> exactly one fine scale mode computed offline
            for ent in edges:
                dofs = dofmap.entity_dofs(1, ent)
                if len(dofs) > 1:
                    # g_sub=const.
                    continue
                try:
                    bcs.update({dofs[0]: 1.})
                except IndexError:
                    # do nothing
                    assert len(dofs) == 0
        else:
            # coarse scale dofs
            coarse_values = interpolate(uD, x_verts)
            for values, ent in zip(coarse_values, vertices):
                dofs = dofmap.entity_dofs(0, ent)
                for k, v in zip(dofs, values):
                    bcs.update({k: v})

            # fine scale dofs
            for ent in edges:
                dofs = dofmap.entity_dofs(1, ent)
                assert len(dofs) <= 1
                try:
                    bcs.update({dofs[0]: 1.})
                except IndexError:
                    # do nothing
                    assert len(dofs) == 0

    for bc in dirichlet["homogeneous"]:
        locator = bc["boundary"]
        uD = bc["value"]
        sub = bc.get("sub")

        # locate entities
        vertices = grid.locate_entities_boundary(0, locator)
        edges = grid.locate_entities_boundary(1, locator)

        if sub is not None:
            assert sub in (0, 1)

            for ent in vertices:
                dofs = dofmap.entity_dofs(0, ent)
                bcs.update({dofs[sub]: 0.0})

            # sanity check
            for ent in edges:
                dofs = dofmap.entity_dofs(1, ent)
                assert dofs
        else:
            for ent in vertices:
                dofs = dofmap.entity_dofs(0, ent)
                for d in dofs:
                    bcs.update({d: 0.0})

            # sanity check
            for ent in edges:
                dofs = dofmap.entity_dofs(1, ent)
                assert not dofs
    return bcs


def get_boundary_dofs(V, marker):
    """get dofs on the boundary"""
    domain = V.mesh
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1
    entities = dolfinx.mesh.locate_entities_boundary(domain, fdim, marker)
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, entities)
    bc = dolfinx.fem.dirichletbc(np.array((0, ) * gdim, dtype=PETSc.ScalarType), dofs)
    dof_indices = bc.dof_indices()[0]
    return dof_indices


class BoundaryDataFactory(object):
    """handles creation of Functions for extension into a domain Ω

    the Functions are constructed such that
        u=g on Γ and u=0 on ∂Ω \\ Γ

    (for the extension also the latter condition is important).
    Usually, we have a bc known as {value, boundary(=Γ)} and
    can create a bc and apply it to a function (dolfinx.fem.petsc.set_bc)
    to achieve the above.
    However, for the extension we require a dirichletbc object that
    applies to the entire boundary ∂Ω.
    """
    def __init__(self, domain, V):
        self.domain = domain
        self.V = V

        # bc handler
        self.bch = BoundaryConditions(domain, V)

        # boundary facets and dofs (entire boundary)
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
        self.boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)

    def create_function_values(self, values, boundary_dofs):
        """create a function and set values

        Parameters
        ----------
        values : np.ndarray
            The values to set.
        boundary_dofs : np.ndarray
            The dof indices of the vector.

        Returns
        -------
        u : dolfinx.fem.Function
        """
        u = dolfinx.fem.Function(self.V)
        u.vector.zeroEntries()
        u.vector.setValues(boundary_dofs, values, addv=PETSc.InsertMode.INSERT)
        u.vector.assemblyBegin()
        u.vector.assemblyEnd()
        return u

    def create_function_bc(self, bc):
        """create a function and set given bc

        Parameters
        ----------
        bc : dict
            A suitable definition of a Dirichlet bc.
            See multi.bcs.BoundaryConditions.add_dirichlet_bc.

        Returns
        -------
        u : dolfinx.fem.Function
        """
        self.bch.clear()
        self.bch.add_dirichlet_bc(**bc)
        bcs = self.bch.bcs
        u = dolfinx.fem.Function(self.V)
        dolfinx.fem.petsc.set_bc(u.vector, bcs)
        u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )
        return u


    def create_bc(self, function):
        """create a bc with value function for the entire boundary

        Parameters
        ----------
        function : dolfinx.fem.Function
            The function to prescribe.

        Returns
        -------
        bc : dolfinx.fem.dirichletbc
        """
        dofs = self.boundary_dofs
        bc = dolfinx.fem.dirichletbc(function, dofs)
        return bc


# adapted version of MechanicsBCs by Thomas Titscher
class BoundaryConditions:
    """Handles dirichlet and neumann boundary conditions

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational domain of the problem.
    space : dolfinx.fem.FunctionSpace
        Finite element space defined on the domain.
    facet_markers : optional, dolfinx.mesh.MeshTags
        The mesh tags defining boundaries.

    """

    def __init__(self, domain, space, facet_markers=None):
        self.domain = domain
        self.V = space

        # create connectivity
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)

        # list of dirichlet boundary conditions
        self._bcs = []

        # handle facets and measure for neumann bcs
        self._neumann_bcs = []
        self._facet_markers = facet_markers
        self._ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers)
        self._v = ufl.TestFunction(space)

    def add_dirichlet_bc(
        self, value, boundary=None, sub=None, method="topological", entity_dim=None
    ):
        """add a Dirichlet BC

        Parameters
        ----------
        value : Function, Constant or np.ndarray or DirichletBCMetaClass
            The Dirichlet function or boundary condition.
        boundary : optional, callable or np.ndarray or int
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
        if boundary is None:
            assert isinstance(value, dolfinx.fem.DirichletBCMetaClass)
            self._bcs.append(value)
        else:
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

                dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim, facets)
            else:
                dofs = dolfinx.fem.locate_dofs_geometrical(V, boundary)

            if isinstance(value, (dolfinx.fem.Constant, np.ndarray, np.float64)):
                bc = dolfinx.fem.dirichletbc(value, dofs, V)
            else:
                try:
                    bc = dolfinx.fem.dirichletbc(value, dofs)
                except AttributeError:
                    f = dolfinx.fem.Function(V)
                    f.interpolate(value)
                    bc = dolfinx.fem.dirichletbc(f, dofs)
            self._bcs.append(bc)

    def add_neumann_bc(self, marker, value):
        """adds a Neumann BC.

        Parameters
        ----------
        marker : int
        value : some ufl type
            The neumann data, e.g. traction vector.

        """
        if isinstance(marker, int):
            assert marker in self._facet_markers.values

        self._neumann_bcs.append([value, marker])

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


def compute_multiscale_bcs(
    problem, cell_index, edge, boundary_data, dofmap, chi=None, product=None, orth=False
):
    """compute multiscale bcs from given boundary data

    The coarse scale basis functions are assumed to be linear functions
    with value 1 or 0 at the endpoints of the edge mesh.
    The dof values for the fine scale edge basis are computed
    via projection of the boundary data onto the fine scale basis.
    If `chi` is None, or the number of dofs per edge (dofmap) is
    < 1, then only the coarse dof values are computed.

    Parameters
    ----------
    problem : multi.problems.LinearProblem
        The linear problem.
    cell_index : int
        The cell index with respect to the coarse grid.
    edge : str
        The boundary edge.
    boundary_data : dolfinx.fem.Function
        The (total) displacement value on the boundary.
        Will be projected onto the edge space.
    dofmap : multi.dofmap.DofMap
        The dofmap of the reduced order model.
    chi : optional, np.ndarray
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
    assert edge in ("bottom", "right", "top", "left")
    local_edge_index = dofmap.dof_layout.local_edge_index_map[edge]
    local_vertices = dofmap.dof_layout.topology[1][local_edge_index]

    edge_space = problem.edge_spaces[edge]
    source = FenicsxVectorSpace(edge_space)

    dofs_per_edge = dofmap.dofs_per_edge
    if isinstance(dofs_per_edge, (int, np.integer)):
        add_fine_bcs = dofs_per_edge > 0
        edge_basis_length = dofs_per_edge
    else:
        assert dofs_per_edge.shape == (dofmap.num_cells, 4)
        dofs_per_edge = dofs_per_edge[cell_index]
        add_fine_bcs = np.sum(dofs_per_edge) > 0
        edge_basis_length = dofs_per_edge[local_edge_index]

    if edge == "bottom":
        component = 0
    elif edge == "top":
        component = 0
    elif edge == "left":
        component = 1
    elif edge == "right":
        component = 1
    else:
        raise NotImplementedError

    # need to know entities wrt the global coarse grid
    # to determine the dof indices of the global problem
    vertices_coarse_cell = dofmap.conn[0].links(cell_index)
    edges_coarse_cell = dofmap.conn[1].links(cell_index)

    boundary_vertices = vertices_coarse_cell[local_vertices]
    boundary_nodes = dolfinx.mesh.compute_midpoints(
        dofmap.grid.mesh, 0, boundary_vertices
    )
    # make sure points are withing problem.domain
    boundary_nodes = np.around(boundary_nodes, decimals=3)

    coarse_values = interpolate(boundary_data, boundary_nodes)
    coarse_dofs = []
    for vertex in boundary_vertices:
        coarse_dofs += dofmap.entity_dofs(0, vertex)

    if not len(coarse_dofs) == coarse_values.size:
        breakpoint()

    # initialize return value
    bcs = {}  # dofs are keys, bc_values are values
    for dof, val in zip(coarse_dofs, coarse_values.flatten()):
        bcs.update({dof: val})

    if add_fine_bcs:
        assert chi is not None
        # ### subtract coarse scale part from boundary data
        # FIXME interpolation for different meshes not supported (yet) in dolfinx
        x_dofs = edge_space.tabulate_dof_coordinates()
        g = interpolate(boundary_data, x_dofs)
        G = source.from_numpy(g.reshape(1, source.dim))

        def boundary(x):
            start = np.isclose(x[component], boundary_nodes[:, component][0])
            end = np.isclose(x[component], boundary_nodes[:, component][1])
            return np.logical_or(start, end)

        line = NumpyLine(boundary_nodes[:, component])
        phi_array = line.interpolate(edge_space, component)
        phi = source.from_numpy(phi_array)
        Gfine = G - phi.lincomb(coarse_values.flatten())

        # ### build inner product for edge space
        boundary_dofs = dolfinx.fem.locate_dofs_geometrical(edge_space, boundary)
        zero = dolfinx.fem.Function(edge_space)
        zero.x.set(0.0)
        product_bc = dolfinx.fem.dirichletbc(zero, boundary_dofs)
        inner_product = InnerProduct(edge_space, product, bcs=(product_bc,))
        matrix = inner_product.assemble_matrix()
        if matrix is not None:
            product = FenicsxMatrixOperator(matrix, edge_space, edge_space)
        else:
            product = None

        edge_basis = source.from_numpy(chi)
        edge_basis = edge_basis[:edge_basis_length]
        if orth:
            coeff = Gfine.inner(edge_basis, product=product)
        else:
            Gramian = edge_basis.gramian(product=product)
            R = edge_basis.inner(Gfine, product=product)
            coeff = np.linalg.solve(Gramian, R)

        # determine entity tag for the edge of the coarse grid cell
        edges_coarse_cell = dofmap.conn[1].links(cell_index)
        edge_tag = edges_coarse_cell[local_edge_index]
        fine_dofs = np.array(dofmap.entity_dofs(1, edge_tag))

        try:
            coeff = coeff.reshape(fine_dofs.shape)
        except ValueError as verr:
            raise ValueError(
                "Number of modes per edge (dofmap) and the number of modes of the edge basis do not agree!"
            ) from verr

        for dof, val in zip(fine_dofs, coeff):
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
