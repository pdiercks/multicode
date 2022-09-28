import numpy as np
from fenics_helpers.boundary import to_floats
import warnings

GMSH_QUADRILATERALS = ("quad", "quad8", "quad9")


def adjacency_graph(cells, cell_type="quad"):
    N = np.unique(cells).size  # number of points
    r = np.zeros((N, N), dtype=int)

    if cell_type == "quad8":
        local_adj = {
            0: (4, 7),
            1: (4, 5),
            2: (5, 6),
            3: (6, 7),
            4: (0, 1),
            5: (1, 2),
            6: (2, 3),
            7: (0, 3),
        }
    elif cell_type == "quad":
        local_adj = {
            0: (1, 3),
            1: (2, 0),
            2: (3, 1),
            3: (0, 2),
        }
    elif cell_type == "line3":
        local_adj = {
            0: (2,),
            1: (2,),
            2: (1, 0),
        }
    else:
        raise NotImplementedError

    for cell in cells:
        for vertex, neighbours in local_adj.items():
            for n in neighbours:
                r[cell[vertex], cell[n]] = 1
    return r


class Quadrilateral:
    """
    v3--e2--v2
    |       |
    e3  f0  e1
    |       |
    v0--e0--v1
    """

    def __init__(self, gmsh_cell_type):
        if gmsh_cell_type not in GMSH_QUADRILATERALS:
            raise AttributeError(
                "Cell type {} is not supported.".format(gmsh_cell_type)
            )
        self.cell_type = gmsh_cell_type
        self.verts = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        self.edges = {}
        self.faces = {}

        if gmsh_cell_type in ("quad8", "quad9"):
            self.edges = {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0)}
            if gmsh_cell_type in ("quad9"):
                self.faces = {0: (0, 1, 2, 3)}

        self.topology = {
            0: {0: (0,), 1: (1,), 2: (2,), 3: (3,)},
            1: self.edges,
            2: self.faces,
        }

    def get_entities(self, dim=None):
        """return entities of given dimension `dim`"""
        if not hasattr(self, "_entities"):
            raise AttributeError("Entities are not set for cell {}".format(type(self)))
        if dim is not None:
            return self._entities[dim]
        return self._entities

    def set_entities(self, gmsh_cell):
        """set entities for current cell `gmsh_cell`"""
        nv = len(self.verts)
        ne = len(self.edges.keys())
        nf = len(self.faces.keys())
        self._entities = {
            0: gmsh_cell[:nv],
            1: gmsh_cell[nv : nv + ne],
            2: gmsh_cell[nv + ne : nv + ne + nf],
        }

    def get_entity_dofs(self):
        if not hasattr(self, "_entity_dofs"):
            raise AttributeError(
                "Entity dofs are not set for cell {}".format(type(self))
            )
        return self._entity_dofs

    def set_entity_dofs(self, dofs_per_vert, dofs_per_edge, dofs_per_face):
        """set local dof indices for each entity of the cell

        Parameters
        ----------
        dofs_per_vert : int
            Number of DoFs per vertex.
        dofs_per_edge : int, np.ndarray
            Number of DoFs per edge. Either an integer value or a numpy array
            of shape ``(4, )``.
        dofs_per_face : int
            Number of DoFs per face.
        """

        counter = 0
        self._entity_dofs = {0: {}, 1: {}, 2: {}}
        for v in range(len(self.verts)):
            self._entity_dofs[0][v] = [
                dofs_per_vert * v + i for i in range(dofs_per_vert)
            ]
        counter += len(self.verts) * dofs_per_vert
        if isinstance(dofs_per_edge, (int, np.integer)):
            for e in range(len(self.edges)):
                self._entity_dofs[1][e] = [
                    counter + dofs_per_edge * e + i for i in range(dofs_per_edge)
                ]
        else:
            for e in range(len(self.edges)):
                self._entity_dofs[1][e] = [
                    counter + dof for dof in range(dofs_per_edge[e])
                ]
                counter += dofs_per_edge[e]
        for f in range(len(self.faces)):
            self._entity_dofs[2][f] = [
                counter + dofs_per_face * f + i for i in range(dofs_per_face)
            ]


class DofMap:
    """class representing a DofMap of a function space where each entity
    (vertex, edge, face in 2d) is associated with the given number of DoFs,
    when calling `distribute_dofs()`.

    Parameters
    ----------
    points : np.ndarray
        The physical coordinates of the nodes of the mesh.
    cells : np.ndarray
        The connectivity of the points.
    cell_type : str, optional
        Gmsh cell type. Supported are `quad`, `quad8` and `quad9`.
    """

    def __init__(self, points, cells, cell_type="quad8"):
        self.points = points
        self.cells = cells
        self._cell = Quadrilateral(cell_type)

        cellpoints = self.points[self.cells[0]]
        self.cell_size = np.around(np.abs(cellpoints[1] - cellpoints[0])[0], decimals=5)

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
        x_dofs = []
        self._dm = {dim: {} for dim in dimension}
        DoF = 0

        if isinstance(dofs_per_edge, (int, np.integer)):
            self._cell.set_entity_dofs(dofs_per_vert, dofs_per_edge, dofs_per_face)
            entity_dofs = self._cell.get_entity_dofs()
            for ci, cell in enumerate(self.cells):
                self._cell.set_entities(cell)
                for dim in dimension:
                    entities = self._cell.get_entities()[dim]
                    for local_ent, ent in enumerate(entities):
                        if ent not in self._dm[dim].keys():
                            self._dm[dim][ent] = []
                            dofs = entity_dofs[dim][local_ent]
                            for dof in dofs:
                                x_dofs.append(self.points[ent])  # store dof coordinates
                                self._dm[dim][ent].append(DoF)
                                DoF += 1
        else:
            assert dofs_per_edge.shape == (len(self.cells), 4)
            for ci, cell in enumerate(self.cells):
                self._cell.set_entity_dofs(
                    dofs_per_vert, dofs_per_edge[ci], dofs_per_face
                )
                entity_dofs = self._cell.get_entity_dofs()
                self._cell.set_entities(cell)
                for dim in dimension:
                    entities = self._cell.get_entities()[dim]
                    for local_ent, ent in enumerate(entities):
                        if ent not in self._dm[dim].keys():
                            self._dm[dim][ent] = []
                            dofs = entity_dofs[dim][local_ent]
                            for dof in dofs:
                                x_dofs.append(self.points[ent])  # store dof coordinates
                                self._dm[dim][ent].append(DoF)
                                DoF += 1
        self.n_dofs = DoF
        self.dofs_per_vert = dofs_per_vert
        self.dofs_per_edge = dofs_per_edge
        self.dofs_per_face = dofs_per_face
        self._x_dofs = np.array(x_dofs)

    def tabulate_dof_coordinates(self):
        """return dof coordinates"""
        if not hasattr(self, "_x_dofs"):
            raise AttributeError("You need to distribute DoFs first")
        return self._x_dofs

    def dofs(self):
        """return total number of dofs"""
        if not hasattr(self, "n_dofs"):
            raise AttributeError("You need to distribute DoFs first")
        return self.n_dofs

    def cell_dofs(self, cell_index):
        """returns dofs for given cell

        Returns
        -------
        dofs : list of int
            The dofs of the given cell.
        """
        if not hasattr(self, "_dm"):
            raise AttributeError("You need to distribute DoFs first")
        dimension = list(range(self.tdim + 1))
        cell = self.cells[cell_index]
        cell_dofs = []
        for dim in dimension:
            self._cell.set_entities(cell)
            entities = self._cell.get_entities()[dim]
            for ent in entities:
                cell_dofs += self._dm[dim][ent]
        return cell_dofs

    def locate_cells(self, X, tol=1e-9):
        """return cell indices for cells containing at least one
        of the points in X

        Parameters
        ----------
        X : list, np.ndarray
            A list of points, where each point is given as list of len(gdim).
        tol : float, optional
            Tolerance used to find coordinate.

        Returns
        -------
        cell_indices : list
            Indices of cells containing given points.
        """
        if isinstance(X, list):
            X = np.array(X).reshape(len(X), self.gdim)
        assert isinstance(X, np.ndarray)

        cell_indices = set()
        for x in X:
            p = np.abs(self.points - x)
            v = np.where(np.all(p < tol, axis=1))[0]
            if v.size < 1:
                warnings.warn(
                    f"The point {x} is not a vertex of the grid.\n"
                    "Looping over cells to determine which cell might contain point.\n"
                    "This may take a while ..."
                )
                for cell_index, cell in enumerate(self.cells):
                    x_cell = self.points[cell][:4]
                    xmin, ymin = x_cell[0]
                    xmax, ymax = x_cell[2]
                    if np.logical_and(xmin <= x[0] <= xmax, ymin <= x[1] <= ymax):
                        cell_indices.add(cell_index)
            else:
                ci = np.where(np.any(np.abs(self.cells - v) < tol, axis=1))[0]
                cell_indices.update(tuple(ci.flatten()))

        return list(cell_indices)

    def locate_dofs(self, X, sub=None, s_=np.s_[:], tol=1e-9):
        """returns dofs at coordinates X

        Parameters
        ----------
        X : list, np.ndarray
            A list of points, where each point is given as list of len(gdim).
        sub : int, optional
            Index of component.
            Note, that this is intended for the case that X contains only vertex
            OR only edge coordinates of the grid!
        s_ : slice, optional
            Return slice of the dofs at each point.
        tol : float, optional
            Tolerance used to find coordinate.

        Returns
        -------
        dofs : np.ndarray
            DoFs at given coordinates.
        """
        if isinstance(X, list):
            X = np.array(X).reshape(len(X), self.gdim)
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X[np.newaxis, :]
            elif X.ndim > 2:
                raise NotImplementedError

        dofs = np.array([], int)
        for x in X:
            p = np.abs(self._x_dofs - x)
            v = np.where(np.all(p < tol, axis=1))[0]
            if v.size < 1:
                raise IndexError(f"The point {x} is not a vertex of the grid!")
            dofs = np.append(dofs, v[s_])

        if sub is not None:
            # FIXME user has to know that things might go wrong if
            # X contains vertex AND edge coordinates ...
            dim = self.gdim
            return dofs[sub::dim]
        else:
            return dofs

    def within_range(self, start, end, tol=1e-6, vertices_only=False, edges_only=False):
        """return all (vertex and edge mid) points within range defined by `start` and `end`.
        Note that a structured grid is assumed for options 'vertices_only' and 'edges_only'.

        Parameters
        ----------
        start : list of float
            The min values of all dimensions within range.
        end : list of float
            The max values of all dimensions within range.
        tol : float, optional
            Tolerance with which points lie within range.
        vertices_only : bool, optional
            If True, return only vertex points.
        edges_only : bool, optional
            If True, return only edge mid points.

        Returns
        -------
        np.ndarray
            Points of mesh within range.
        """
        points = self.points
        cell_size = self.cell_size

        start = to_floats(start)
        end = to_floats(end)

        # this code block is part of fenics_helpers.boundary.within_range
        # adjust the values such that start < end for all dimensions
        assert len(start) == len(end)
        for i in range(len(start)):
            if start[i] > end[i]:
                start[i], end[i] = end[i], start[i]

        within_range = np.where(
            np.logical_and(points[:, 0] + tol >= start[0], points[:, 0] - tol <= end[0])
        )[0]
        for i in range(1, self.gdim):
            ind = np.where(
                np.logical_and(
                    points[:, i] + tol >= start[i], points[:, i] - tol <= end[i]
                )
            )[0]
            within_range = np.intersect1d(within_range, ind)

        if vertices_only:
            p = points[within_range]
            mask = np.mod(np.around(p, decimals=5), cell_size)
            return p[np.all(mask == 0, axis=1)]

        if edges_only:
            p = points[within_range]
            mask = np.mod(np.around(p, decimals=5), cell_size)
            return p[np.invert(np.all(mask == 0, axis=1))]

        return points[within_range]

    # parts of the code copied from fenics_helpers.boundary.plane_at
    def plane_at(
        self, coordinate, dim=0, tol=1e-9, vertices_only=False, edges_only=False
    ):
        """return all (vertex and edge mid) points in plane where dim equals coordinate
        Note that a structured grid is assumed for options 'vertices_only' and 'edges_only'.

        Parameters
        ----------
        coordinate : float
            The coordinate.
        dim : int, str, optional
            The spatial dimension.
        tol : float, optional
            Tolerance with which points match coordinate.
        vertices_only : bool, optional
            If True, return only vertex points.
        edges_only : bool, optional
            If True, return only edge mid points.

        Returns
        -------
        np.ndarray
            Points of mesh in given plane.
        """
        cell_size = self.cell_size

        if dim in ["x", "X"]:
            dim = 0
        if dim in ["y", "Y"]:
            dim = 1
        if dim in ["z", "Z"]:
            dim = 2

        assert dim in [0, 1, 2]
        p = self.points[np.where(np.abs(self.points[:, dim] - coordinate) < tol)[0]]

        if vertices_only:
            mask = np.mod(np.around(p, decimals=5), cell_size)
            return p[np.all(mask == 0, axis=1)]
        if edges_only:
            mask = np.mod(np.around(p, decimals=5), cell_size)
            return p[np.invert(np.all(mask == 0, axis=1))]
        return p

    def get_cell_points(self, cells, gmsh_nodes=None):
        """returns point coordinates for given cells

        Parameters
        ----------
        cells : list of int
            The cells for which points should be returned.
        gmsh_nodes : list, np.ndarray, optional
            Restrict return value to a subset of all nodes.
            Gmsh node ordering is used.

        Returns
        -------
        p : np.ndarray
            The point coordinates.

        """
        n_nodes = self.cells.shape[1]
        if gmsh_nodes is None:
            gmsh_nodes = np.arange(n_nodes)
        else:
            if isinstance(gmsh_nodes, list):
                gmsh_nodes = np.array(gmsh_nodes)
            assert not gmsh_nodes.size > n_nodes

        # return unique nodes keeping ordering
        nodes = self.cells[np.ix_(cells, gmsh_nodes)].flatten()
        _, idx = np.unique(nodes, return_index=True)

        return self.points[nodes[np.sort(idx)]]
