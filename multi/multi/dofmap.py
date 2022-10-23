import dolfinx
import numpy as np
import basix


class QuadrilateralDofLayout(object):
    """
    NOTE
    this is mainly tested and implemented to be used to
    construct a custom DofMap on quadrilateral meshes

    quadrilateral reference cell topology

    FIXME
    currently DofMap relies on domain.topology to distribute the dofs
    therefore, the `dofs_per_edge` need to be given in the local ordering
    compliant with domain.topology
    Ideally, this local ordering of the dofs should follow the basix element dof layout.

    Order (1)
    ---------
    v2--e3--v3
    |       |
    e1  f0  e2
    |       |
    v0--e0--v1

    see basix.geometry(basix.CellType.quadrilateral)
    and basix.topology(basix.CellType.quadrilateral)

    BUT for now the following ordering of vertices and edges is assumed:

    Order (2)
    ---------
    v1--e2--v3
    |       |
    e0  f0  e3
    |       |
    v0--e1--v2

    EDIT 12.10.2022
    Apparently the built-in mesh has above ordering (2) and
    a mesh read from msh has ordering (1) ...
    --> decision: go with ordering (1) and only use Gmsh, but
    not the built-in meshes with DofMap, StructuredQuadGrid, etc.

    """

    def __init__(self):
        self.topology = basix.topology(basix.CellType.quadrilateral)
        self.geometry = basix.geometry(basix.CellType.quadrilateral)
        self.num_entities = [len(ents) for ents in self.topology]
        self.local_edge_index_map = {"left": 1, "bottom": 0, "top": 3, "right": 2}

    def get_entity_dofs(self):
        return self.__entity_dofs

    def set_entity_dofs(self, ndofs_per_ent):
        """set number of dofs per entity

        Parameters
        ----------
        ndofs_per_ent : tuple of int or np.ndarray
            Number of dofs per entity. For the edges this can be
            a numpy array otherwise an integer value is allowed.
        """
        assert len(ndofs_per_ent) == len(self.num_entities)

        self.__entity_dofs = {0: {}, 1: {}, 2: {}}  # key=entity_dim, value=dict
        counter = 0
        for dim, ndofs in enumerate(ndofs_per_ent):
            num_entities = self.num_entities[dim]
            if isinstance(ndofs, int):
                ndofs = [ndofs for _ in range(num_entities)]
            for entity in range(num_entities):
                self.__entity_dofs[dim][entity] = [
                    counter + dof for dof in range(ndofs[entity])
                ]
                counter += ndofs[entity]


# FIXME
# currently DofMap relies on domain.topology to distribute the dofs
# therefore, the `dofs_per_edge` need to be given in the local ordering
# compliant with domain.topology
# Ideally, this local ordering of the dofs should follow the basix element dof layout
class DofMap:
    """class representing a DofMap of a function space where each entity
    (vertex, edge, face in 2d) is associated with the given number of DoFs,
    when calling `distribute_dofs()`.

    Parameters
    ----------
    grid : multi.domain.StructuredQuadGrid
        The quadrilateral mesh of the computational domain.
    """

    def __init__(self, grid):
        self.grid = grid
        self.dof_layout = QuadrilateralDofLayout()

        # create connectivities
        self.conn = []
        domain = grid.mesh
        for dim in range(len(self.dof_layout.num_entities)):
            domain.topology.create_connectivity(2, dim)
            self.conn.append(domain.topology.connectivity(2, dim))
        self.num_cells = self.conn[2].num_nodes

    def distribute_dofs(self, dofs_per_vert, dofs_per_edge, dofs_per_face=0):
        """set number of DoFs per entity and distribute dofs

        Parameters
        ----------
        dofs_per_vert : int
            Number of DoFs per vertex.
        dofs_per_edge : int, np.ndarray
            Number of DoFs per edge. This can be an integer value to set number
            of DoFs for each edge to the same value or an array of shape
            ``(len(self.cells), 4)`` to set number of DoFs for each cell and
            its four edges individually.
        dofs_per_face : int, optional
            Number of DoFs per face.
        """
        # ### initialize
        # there are only vertices, edges and faces for quadrilaterals
        dimension = [0, 1, 2]
        self._dm = {dim: {} for dim in dimension}
        DoF = 0

        num_cells = self.num_cells

        if isinstance(dofs_per_edge, (int, np.integer)):
            dofs_per_edge = np.ones((num_cells, 4), dtype=np.intc) * dofs_per_edge
        else:
            assert dofs_per_edge.shape == (num_cells, 4)

        for cell_index in range(num_cells):
            self.dof_layout.set_entity_dofs(
                (dofs_per_vert, dofs_per_edge[cell_index], dofs_per_face)
            )
            entity_dofs = self.dof_layout.get_entity_dofs()
            for dim, conn in enumerate(self.conn):
                entities = conn.links(cell_index)
                for local_ent, ent in enumerate(entities):
                    if ent not in self._dm[dim].keys():
                        self._dm[dim][ent] = []
                        dofs = entity_dofs[dim][local_ent]
                        for dof in dofs:
                            self._dm[dim][ent].append(DoF)
                            DoF += 1

        self._n_dofs = DoF
        self.dofs_per_vert = dofs_per_vert
        self.dofs_per_edge = dofs_per_edge
        self.dofs_per_face = dofs_per_face

    # FIXME
    # pass StructuredQuadGrid to Dofmap instead of the dolfinx.mesh.Mesh
    # StructuredQuadGrid.get_entities(dim, cell_index) works topologically whereas
    # Dofmap.get_entities(dim, marker) determines entities geometrically !!
    # TODO StructuredQuadGrid as arg to DofMap
    # TODO add methods locate_entities and locate_entities_boundary to StructuredQuadGrid
    # def get_entities(self, dim, marker):
    #     """dolfinx.mesh.locate_entities"""
    #     return dolfinx.mesh.locate_entities(self.domain, dim, marker)

    # def get_entities_boundary(self, dim, marker):
    #     """dolfinx.mesh.locate_entities_boundary"""
    #     assert dim < self.domain.topology.dim
    #     return dolfinx.mesh.locate_entities_boundary(self.domain, dim, marker)

    def get_entity_coordinates(self, dim, entities):
        """return coordinates of `entities` of dimension `dim`"""
        return dolfinx.mesh.compute_midpoints(self.domain, dim, entities)

    def num_dofs(self):
        """return total number of dofs"""
        if not hasattr(self, "_n_dofs"):
            raise AttributeError("You need to distribute DoFs first")
        return self._n_dofs

    def cell_dofs(self, cell_index):
        """returns dofs for given cell

        Returns
        -------
        dofs : list of int
            The dofs of the given cell.
        """
        if not hasattr(self, "_dm"):
            raise AttributeError("You need to distribute DoFs first")

        num_cells = self.num_cells
        assert cell_index in np.arange(num_cells)

        cell_dofs = []
        for dim, conn in enumerate(self.conn):
            entities = conn.links(cell_index)
            for ent in entities:
                cell_dofs += self._dm[dim][ent]
        return cell_dofs

    def entity_dofs(self, dim, entity):
        """return all dofs for entity `entity` of dimension `dim`"""
        return self._dm[dim][entity]

    # FIXME where is this actually needed?
    # could build V = dolfinx.FunctionSpace(domain, ("Lagrange", 2)) since
    # the VectorFunctionSpace also returns each dof coordinate only once
    # def tabulate_dof_coordinates(self):
    #     """return dof coordinates"""
    #     if not hasattr(self, "_x_dofs"):
    #         raise AttributeError("You need to distribute DoFs first")
    #     return self._x_dofs

    # def locate_cells(self, X, tol=1e-9):
    #     """return cell indices for cells containing at least one
    #     of the points in X

    #     Parameters
    #     ----------
    #     X : list, np.ndarray
    #         A list of points, where each point is given as list of len(gdim).
    #     tol : float, optional
    #         Tolerance used to find coordinate.

    #     Returns
    #     -------
    #     cell_indices : list
    #         Indices of cells containing given points.
    #     """
    #     if isinstance(X, list):
    #         X = np.array(X).reshape(len(X), self.gdim)
    #     assert isinstance(X, np.ndarray)

    #     cell_indices = set()
    #     for x in X:
    #         p = np.abs(self.points - x)
    #         v = np.where(np.all(p < tol, axis=1))[0]
    #         if v.size < 1:
    #             warnings.warn(
    #                 f"The point {x} is not a vertex of the grid.\n"
    #                 "Looping over cells to determine which cell might contain point.\n"
    #                 "This may take a while ..."
    #             )
    #             for cell_index, cell in enumerate(self.cells):
    #                 x_cell = self.points[cell][:4]
    #                 xmin, ymin = x_cell[0]
    #                 xmax, ymax = x_cell[2]
    #                 if np.logical_and(xmin <= x[0] <= xmax, ymin <= x[1] <= ymax):
    #                     cell_indices.add(cell_index)
    #         else:
    #             ci = np.where(np.any(np.abs(self.cells - v) < tol, axis=1))[0]
    #             cell_indices.update(tuple(ci.flatten()))

    #     return list(cell_indices)

    # FIXME self._x_dofs required
    # def locate_dofs(self, X, sub=None, s_=np.s_[:], tol=1e-9):
    #     """returns dofs at coordinates X

    #     Parameters
    #     ----------
    #     X : list, np.ndarray
    #         A list of points, where each point is given as list of len(gdim).
    #     sub : int, optional
    #         Index of component.
    #         Note, that this is intended for the case that X contains only vertex
    #         OR only edge coordinates of the grid!
    #     s_ : slice, optional
    #         Return slice of the dofs at each point.
    #     tol : float, optional
    #         Tolerance used to find coordinate.

    #     Returns
    #     -------
    #     dofs : np.ndarray
    #         DoFs at given coordinates.
    #     """
    #     if isinstance(X, list):
    #         X = np.array(X).reshape(len(X), self.gdim)
    #     elif isinstance(X, np.ndarray):
    #         if X.ndim == 1:
    #             X = X[np.newaxis, :]
    #         elif X.ndim > 2:
    #             raise NotImplementedError

    #     dofs = np.array([], int)
    #     for x in X:
    #         p = np.abs(self._x_dofs - x)
    #         v = np.where(np.all(p < tol, axis=1))[0]
    #         if v.size < 1:
    #             raise IndexError(f"The point {x} is not a vertex of the grid!")
    #         dofs = np.append(dofs, v[s_])

    #     if sub is not None:
    #         # FIXME user has to know that things might go wrong if
    #         # X contains vertex AND edge coordinates ...
    #         dim = self.gdim
    #         return dofs[sub::dim]
    #     else:
    #         return dofs

    # def within_range(self, start, end, tol=1e-6, vertices_only=False, edges_only=False):
    #     """return all (vertex and edge mid) points within range defined by `start` and `end`.
    #     Note that a structured grid is assumed for options 'vertices_only' and 'edges_only'.

    #     Parameters
    #     ----------
    #     start : list of float
    #         The min values of all dimensions within range.
    #     end : list of float
    #         The max values of all dimensions within range.
    #     tol : float, optional
    #         Tolerance with which points lie within range.
    #     vertices_only : bool, optional
    #         If True, return only vertex points.
    #     edges_only : bool, optional
    #         If True, return only edge mid points.

    #     Returns
    #     -------
    #     np.ndarray
    #         Points of mesh within range.
    #     """
    #     points = self.points
    #     cell_size = self.cell_size

    #     start = to_floats(start)
    #     end = to_floats(end)

    #     # this code block is part of fenics_helpers.boundary.within_range
    #     # adjust the values such that start < end for all dimensions
    #     assert len(start) == len(end)
    #     for i in range(len(start)):
    #         if start[i] > end[i]:
    #             start[i], end[i] = end[i], start[i]

    #     within_range = np.where(
    #         np.logical_and(points[:, 0] + tol >= start[0], points[:, 0] - tol <= end[0])
    #     )[0]
    #     for i in range(1, self.gdim):
    #         ind = np.where(
    #             np.logical_and(
    #                 points[:, i] + tol >= start[i], points[:, i] - tol <= end[i]
    #             )
    #         )[0]
    #         within_range = np.intersect1d(within_range, ind)

    #     if vertices_only:
    #         p = points[within_range]
    #         mask = np.mod(np.around(p, decimals=5), cell_size)
    #         return p[np.all(mask == 0, axis=1)]

    #     if edges_only:
    #         p = points[within_range]
    #         mask = np.mod(np.around(p, decimals=5), cell_size)
    #         return p[np.invert(np.all(mask == 0, axis=1))]

    #     return points[within_range]

    # parts of the code copied from fenics_helpers.boundary.plane_at
    # def plane_at(
    #     self, coordinate, dim=0, tol=1e-9, vertices_only=False, edges_only=False
    # ):
    #     """return all (vertex and edge mid) points in plane where dim equals coordinate
    #     Note that a structured grid is assumed for options 'vertices_only' and 'edges_only'.

    #     Parameters
    #     ----------
    #     coordinate : float
    #         The coordinate.
    #     dim : int, str, optional
    #         The spatial dimension.
    #     tol : float, optional
    #         Tolerance with which points match coordinate.
    #     vertices_only : bool, optional
    #         If True, return only vertex points.
    #     edges_only : bool, optional
    #         If True, return only edge mid points.

    #     Returns
    #     -------
    #     np.ndarray
    #         Points of mesh in given plane.
    #     """
    #     cell_size = self.cell_size

    #     if dim in ["x", "X"]:
    #         dim = 0
    #     if dim in ["y", "Y"]:
    #         dim = 1
    #     if dim in ["z", "Z"]:
    #         dim = 2

    #     assert dim in [0, 1, 2]
    #     p = self.points[np.where(np.abs(self.points[:, dim] - coordinate) < tol)[0]]

    #     if vertices_only:
    #         mask = np.mod(np.around(p, decimals=5), cell_size)
    #         return p[np.all(mask == 0, axis=1)]
    #     if edges_only:
    #         mask = np.mod(np.around(p, decimals=5), cell_size)
    #         return p[np.invert(np.all(mask == 0, axis=1))]
    #     return p

    # def get_cell_points(self, cells, gmsh_nodes=None):
    #     """returns point coordinates for given cells

    #     Parameters
    #     ----------
    #     cells : list of int
    #         The cells for which points should be returned.
    #     gmsh_nodes : list, np.ndarray, optional
    #         Restrict return value to a subset of all nodes.
    #         Gmsh node ordering is used.

    #     Returns
    #     -------
    #     p : np.ndarray
    #         The point coordinates.

    #     """
    #     n_nodes = self.cells.shape[1]
    #     if gmsh_nodes is None:
    #         gmsh_nodes = np.arange(n_nodes)
    #     else:
    #         if isinstance(gmsh_nodes, list):
    #             gmsh_nodes = np.array(gmsh_nodes)
    #         assert not gmsh_nodes.size > n_nodes

    #     # return unique nodes keeping ordering
    #     nodes = self.cells[np.ix_(cells, gmsh_nodes)].flatten()
    #     _, idx = np.unique(nodes, return_index=True)

    #     return self.points[nodes[np.sort(idx)]]
