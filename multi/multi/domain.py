import os

# from pathlib import Path

import gmsh
import tempfile
import meshio

# import dolfin as df
import numpy as np

# from fenics_helpers.boundary import to_floats
from multi.common import to_floats

# from multi.dofmap import Quadrilateral

GMSH_QUADRILATERALS = ("quad", "quad8", "quad9")


class Domain(object):
    """Class to represent a computational domain

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The partition of the domain.
    cell_markers : TODO
    facet_markers : TODO
    index : optional, int
        The identification number of the domain.
    """

    def __init__(self, mesh, cell_markers=None, facet_markers=None, index=None):
        self.mesh = mesh
        self._x = mesh.geometry.x
        self.cell_markers = cell_markers
        self.facet_markers = facet_markers
        self.index = index

    def translate(self, dx):
        dx = np.array(dx)
        self._x += dx

    # TODO parallel?
    def xmin(self):
        return np.amin(self._x, axis=0)

    # TODO parallel?
    def xmax(self):
        return np.amax(self._x, axis=0)

# class Domain(object):
#     """Class to represent a computational domain

#     Parameters
#     ----------
#     mesh : str, dolfin.cpp.mesh.Mesh
#         The discretization of the subdomain given as XMDF file (incl. ext) to load or
#         instance of dolfin.cpp.mesh.Mesh.
#     _id : int
#         The identification number of the domain.
#     subdomains : optional
#         Set to True if domain has subdomains represented by a df.MeshFunction stored in
#         the XDMF file given as `mesh` or provide df.MeshFunction directly.

#     """

#     def __init__(self, mesh, _id=1, subdomains=None):
#         if isinstance(mesh, df.cpp.mesh.Mesh):
#             self.mesh = mesh
#             self.subdomains = subdomains
#         else:
#             self.xdmf_file = Path(mesh)
#             self.mesh = df.Mesh()
#             mvc = df.MeshValueCollection("size_t", self.mesh, dim=None)
#             # note that mvc.dim() will be overwritten with
#             # self.mesh.topology().dim() by f.read(mvc, ...) below

#             with df.XDMFFile(self.xdmf_file.as_posix()) as f:
#                 f.read(self.mesh)
#                 if subdomains:
#                     f.read(mvc, "gmsh:physical")

#             if subdomains:
#                 self.subdomains = df.MeshFunction("size_t", self.mesh, mvc)
#             else:
#                 self.subdomains = subdomains
#         self._id = int(_id)
#         self.gdim = self.mesh.geometric_dimension()
#         self.tdim = self.mesh.topology().dim()

#     def translate(self, point):
#         """translate the domain in space

#         Parameters
#         ----------
#         point : dolfin.Point
#             The point by which to translate.

#         Note: if `self.edges` evaluates to True, edge
#         meshes are translated as well.
#         """
#         self.mesh.translate(point)

#     @property
#     def xmin(self):
#         return self.mesh.coordinates()[:, 0].min()

#     @property
#     def xmax(self):
#         return self.mesh.coordinates()[:, 0].max()

#     @property
#     def ymin(self):
#         try:
#             v = self.mesh.coordinates()[:, 1].min()
#         except IndexError:
#             v = 0.0
#         return v

#     @property
#     def ymax(self):
#         try:
#             v = self.mesh.coordinates()[:, 1].max()
#         except IndexError:
#             v = 0.0
#         return v

#     @property
#     def zmin(self):
#         try:
#             v = self.mesh.coordinates()[:, 2].min()
#         except IndexError:
#             v = 0.0
#         return v

#     @property
#     def zmax(self):
#         try:
#             v = self.mesh.coordinates()[:, 2].max()
#         except IndexError:
#             v = 0.0
#         return v


# class RectangularDomain(Domain):
#     """
#     Parameters
#     ----------
#     mesh : str, dolfin.cpp.mesh.Mesh
#         The discretization of the subdomain given as XMDF file (incl. ext) to load or
#         instance of dolfin.cpp.mesh.Mesh.
#     _id : int
#         The identification number of the domain.
#     subdomains : optional
#         Set to True if domain has subdomains represented by a df.MeshFunction stored in
#         the XDMF file given as `mesh` or provide df.MeshFunction directly.
#     edges : bool, optional
#         If True read mesh for each edge (boundary) of the mesh.
#     """

#     def __init__(self, mesh, _id=1, subdomains=None, edges=False):
#         super().__init__(mesh, _id, subdomains)
#         if edges:
#             self._read_edges()
#         else:
#             self.edges = False

#     def _read_edges(self):
#         """reads meshes assuming `mesh` was a xdmf file and edge meshes
#         are present in the same directory"""
#         path = os.path.dirname(os.path.abspath(self.xdmf_file))
#         base = os.path.splitext(os.path.basename(self.xdmf_file))[0]
#         ext = os.path.splitext(os.path.basename(self.xdmf_file))[1]

#         def read(xdmf):
#             mesh = df.Mesh()
#             with df.XDMFFile(xdmf) as f:
#                 f.read(mesh)
#             return mesh

#         edge_meshes = []
#         boundary = ["bottom", "right", "top", "left"]
#         for b in boundary:
#             edge = path + "/" + base + f"_{b}" + ext
#             mesh = read(edge)
#             edge_meshes.append(mesh)
#         self.edges = tuple(edge_meshes)

#     def get_nodes(self, n=4):
#         """get nodes of the rectangular domain

#         Parameters
#         ----------
#         n : int, optional
#             Number of nodes.

#         """

#         def midpoint(a, b):
#             return a + (b - a) / 2

#         nodes = np.array(
#             [
#                 [self.xmin, self.ymin],
#                 [self.xmax, self.ymin],
#                 [self.xmax, self.ymax],
#                 [self.xmin, self.ymax],
#                 [midpoint(self.xmin, self.xmax), self.ymin],
#                 [self.xmax, midpoint(self.ymin, self.ymax)],
#                 [midpoint(self.xmin, self.xmax), self.ymax],
#                 [self.xmin, midpoint(self.ymin, self.ymax)],
#                 [midpoint(self.xmin, self.xmax), midpoint(self.ymin, self.ymax)],
#             ]
#         )
#         return nodes[:n]

#     def translate(self, point):
#         """translate the domain in space

#         Parameters
#         ----------
#         point : dolfin.Point
#             The point by which to translate.

#         Note: if `self.edges` evaluates to True, edge
#         meshes are translated as well.
#         """
#         self.mesh.translate(point)
#         # update edges if True
#         if self.edges:
#             for edge in self.edges:
#                 edge.translate(point)


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


# TODO args: mesh, cell_markers, facet_markers
class StructuredGrid(object):
    """class representing a structured (coarse scale) grid

    Each coarse cell is associated with a fine scale grid which
    needs to be set through `self.fine_grids`.

    Parameters
    ----------
    points : np.ndarray
        The physical coordinates of the nodes of the mesh.
    cells : np.ndarray
        The connectivity of the points.
    tdim : int
        The topological dimension of the grid.
    cell_type : str, optional
        Gmsh cell type. Supported are `quad`, `quad8` and `quad9`.
    """

    def __init__(self, points, cells, tdim, cell_type="quad"):
        self.points = points
        self.cells = cells
        self.tdim = tdim
        self.gdim = points.shape[1]
        x = points[cells[0]]
        self.cell_size = np.around(np.abs(x[1] - x[0])[0], decimals=5)
        self._cell = Quadrilateral(cell_type)

    @property
    def cell_sets(self):
        return self._cell_sets

    @cell_sets.setter
    def cell_sets(self, pairs):
        """set cell sets for given pairs of key and array of cell indices"""
        self._cell_sets = pairs

    def get_patch(self, cell_index, layer=1):
        # TODO test for structured and unstructured mesh of different cell types
        n = cell_index
        for _ in range(layer):
            p = self.cells[n]
            p.shape = (p.size, -1)
            contains_p = np.subtract(self.cells, p[:, np.newaxis])
            neighbours = np.where(np.abs(contains_p) < 1e-6)[1]
            n = np.unique(neighbours)

        return n

    def get_point_tags(self, coord, tol=1e-6):
        """get point tags for given coordinates"""
        assert coord.shape[1] == self.gdim

        tags = np.array([], int)
        for x in coord:
            p = np.abs(self.points - x)
            v = np.where(np.all(p < tol, axis=1))[0]
            if v.size < 1:
                raise IndexError(f"The point {x} is not a vertex of the grid!")
            tags = np.append(tags, v)

        return tags

    def get_cells_by_points(self, points):
        """get all cells that contain the given point tags

        Parameters
        ----------
        points : np.ndarray
            The global tags of the points.

        Returns
        -------
        cells : np.ndarray
            The cell indices.
        """
        p = points.copy()
        p.shape = (p.size, -1)
        contains_p = np.subtract(self.cells, p[:, np.newaxis])
        neighbours = np.where(np.abs(contains_p) < 1e-6)[1]
        cells = np.unique(neighbours)
        return cells

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

    def get_points_by_cells(self, cells, gmsh_nodes=None):
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

    @property
    def fine_grids(self):
        return self._fine_grids

    @fine_grids.setter
    def fine_grids(self, values):
        """values as array of length (num_cells,) holding path to fine grid"""
        # TODO only support .msh format for fine grids of this class
        self._fine_grids = values

    def create_fine_grid(self, cells, output):
        """creates a fine scale grid for given cells"""
        # cases: (a) single cell, (b) patch of cells, (c) entire coarse grid

        # initialize
        subdomains = []

        active_cells = self.cells[cells]
        fine_grids = self.fine_grids[cells]

        for cell, grid_path in zip(active_cells, fine_grids):
            points = np.around(self.points[cell], decimals=5)

            mesh = meshio.read(grid_path)

            # translation
            mesh.points += points[0]

            with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tf:
                subdomains.append(tf.name)
                meshio.write(tf.name, mesh, file_format="gmsh")
                print(tf.name)

        # merge subdomains
        gmsh.initialize()
        gmsh.clear()
        gmsh.model.add("fine_grid")

        for msh_file in subdomains:
            gmsh.merge(msh_file)
        gmsh.model.geo.remove_all_duplicates()
        gmsh.model.mesh.remove_duplicate_nodes()
        gmsh.model.mesh.remove_duplicate_elements()

        gmsh.model.mesh.generate(self.tdim)
        gmsh.write(output)
        gmsh.finalize()

        # clean up
        for msh_file in subdomains:
            os.remove(msh_file)
