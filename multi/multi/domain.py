import os
import gmsh
import tempfile
import meshio
import numpy as np
import dolfinx


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
        self.cell_markers = cell_markers
        self.facet_markers = facet_markers
        self.index = index
        self._x = mesh.geometry.x

    def translate(self, dx):
        dx = np.array(dx)
        self._x += dx

    @property
    def xmin(self):
        return np.amin(self._x, axis=0)

    @property
    def xmax(self):
        return np.amax(self._x, axis=0)


class RectangularDomain(Domain):
    """
    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The partition of a rectangular domain.
    cell_markers : optional
    facet_markers : optional
    index : optional, int
        The identification number of the domain.
    edges : optional, dict
        Partitions of the four edges of the rectangular domain.
        Keys: ['bottom', 'right', 'top', 'left'], Values: dolfinx.mesh.Mesh.
    """

    def __init__(self, mesh, cell_markers=None, facet_markers=None, index=None, edges=None):
        super().__init__(mesh, cell_markers, facet_markers, index)
        self.edges = edges

    def get_nodes(self, n=4):
        """get nodes of the rectangular domain

        Parameters
        ----------
        n : int, optional
            Number of nodes.

        """

        def midpoint(a, b):
            return a + (b - a) / 2

        xmin, ymin, zmin = self.xmin
        xmax, ymax, zmax = self.xmax

        # return nodes in same order as multi.dofmap.CellDofLayout
        nodes = np.array(
            [
                [xmin, ymin, zmin],
                [xmin, ymax, zmin],
                [xmax, ymin, zmin],
                [xmax, ymax, zmin],
                [xmin, midpoint(ymin, ymax), zmin],
                [midpoint(xmin, xmax), ymin, zmin],
                [midpoint(xmin, xmax), ymax, zmin],
                [xmax, midpoint(ymin, ymax), zmin],
                [midpoint(xmin, xmax), midpoint(ymin, ymax), zmin],
            ]
        )
        return nodes[:n]

    def translate(self, dx):
        """translate the domain in space

        Parameters
        ----------
        point : dolfin.Point
            The point by which to translate.

        Note: if `self.edges` evaluates to True, edge
        meshes are translated as well.
        """
        dx = np.array(dx)
        self._x += dx
        # update edges if True
        if self.edges:
            for edge in self.edges.values():
                xg = edge.geometry.x
                xg += dx


class StructuredQuadGrid(object):
    """class representing a structured (coarse scale) quadrilateral grid

    Each coarse quadrilateral cell is associated with a fine scale grid which
    needs to be set through `self.fine_grids`.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The partition of the domain.
    cell_markers : TODO
    facet_markers : TODO
    """

    def __init__(self, mesh, cell_markers=None, facet_markers=None):
        self.mesh = mesh
        self.cell_markers = cell_markers
        self.facet_markers = facet_markers

        # bounding box tree
        self.bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)

        self.mesh.topology.create_connectivity(2, 0)
        self.mesh.topology.create_connectivity(2, 1)
        self.mesh.topology.create_connectivity(0, 2)

    @property
    def cell_sets(self):
        return self._cell_sets

    @cell_sets.setter
    def cell_sets(self, pairs):
        """set cell sets for given pairs of key and array of cell indices"""
        self._cell_sets = pairs

    def get_patch(self, cell_index):
        """return all cells neighbouring cell with index `cell_index`"""
        point_tags = self.get_cell_entities(cell_index, 0)
        conn_02 = self.mesh.topology.connectivity(0, 2)
        cells = list()
        for tag in point_tags:
            cells.append(conn_02.links(tag))
        return np.unique(np.hstack(cells))

    def get_cells_points(self, x):
        """return all cells containing points given by coordinates `x`"""
        try:
            x = x.reshape(int(x.size/3), 3)
        except ValueError as err:
            raise err("x.shape = (num_points, 3) is required!")

        # Find cells whose bounding-box collide with the the points
        bb_tree = self.bb_tree
        cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, x)
        assert cell_candidates.num_nodes < 2
        return cell_candidates.links(0)

    def get_cells_point_tags(self, tags):
        """return all cells containing points given by the point tags `tags`"""
        conn_02 = self.mesh.topology.connectivity(0, 2)
        cells = list()
        for tag in tags.flatten():
            c = conn_02.links(tag)
            cells.append(c)
        return np.unique(cells)

    def get_cell_entities(self, cell_index, dim):
        """get entities of dimension `dim` for cell with index `cell_index`"""
        assert dim in (0, 1)
        conn = self.mesh.topology.connectivity(2, dim)
        return conn.links(cell_index)

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
