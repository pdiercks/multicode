import os
from pathlib import Path

import gmsh
import tempfile
import meshio
import dolfin as df
import numpy as np


class Domain(object):
    """Class to represent a computational domain

    Parameters
    ----------
    mesh : str, dolfin.cpp.mesh.Mesh
        The discretization of the subdomain given as XMDF file (incl. ext) to load or
        instance of dolfin.cpp.mesh.Mesh.
    _id : int
        The identification number of the domain.
    subdomains : optional
        Set to True if domain has subdomains represented by a df.MeshFunction stored in
        the XDMF file given as `mesh` or provide df.MeshFunction directly.

    """

    def __init__(self, mesh, _id=1, subdomains=None):
        if isinstance(mesh, df.cpp.mesh.Mesh):
            self.mesh = mesh
            self.subdomains = subdomains
        else:
            self.xdmf_file = Path(mesh)
            self.mesh = df.Mesh()
            mvc = df.MeshValueCollection("size_t", self.mesh, dim=None)
            # note that mvc.dim() will be overwritten with
            # self.mesh.topology().dim() by f.read(mvc, ...) below

            with df.XDMFFile(self.xdmf_file.as_posix()) as f:
                f.read(self.mesh)
                if subdomains:
                    f.read(mvc, "gmsh:physical")

            if subdomains:
                self.subdomains = df.MeshFunction("size_t", self.mesh, mvc)
            else:
                self.subdomains = subdomains
        self._id = int(_id)
        self.gdim = self.mesh.geometric_dimension()
        self.tdim = self.mesh.topology().dim()

    def translate(self, point):
        """translate the domain in space

        Parameters
        ----------
        point : dolfin.Point
            The point by which to translate.

        Note: if `self.edges` evaluates to True, edge
        meshes are translated as well.
        """
        self.mesh.translate(point)

    @property
    def xmin(self):
        return self.mesh.coordinates()[:, 0].min()

    @property
    def xmax(self):
        return self.mesh.coordinates()[:, 0].max()

    @property
    def ymin(self):
        try:
            v = self.mesh.coordinates()[:, 1].min()
        except IndexError:
            v = 0.0
        return v

    @property
    def ymax(self):
        try:
            v = self.mesh.coordinates()[:, 1].max()
        except IndexError:
            v = 0.0
        return v

    @property
    def zmin(self):
        try:
            v = self.mesh.coordinates()[:, 2].min()
        except IndexError:
            v = 0.0
        return v

    @property
    def zmax(self):
        try:
            v = self.mesh.coordinates()[:, 2].max()
        except IndexError:
            v = 0.0
        return v


class RectangularDomain(Domain):
    """
    Parameters
    ----------
    mesh : str, dolfin.cpp.mesh.Mesh
        The discretization of the subdomain given as XMDF file (incl. ext) to load or
        instance of dolfin.cpp.mesh.Mesh.
    _id : int
        The identification number of the domain.
    subdomains : optional
        Set to True if domain has subdomains represented by a df.MeshFunction stored in
        the XDMF file given as `mesh` or provide df.MeshFunction directly.
    edges : bool, optional
        If True read mesh for each edge (boundary) of the mesh.
    """

    def __init__(self, mesh, _id=1, subdomains=None, edges=False):
        super().__init__(mesh, _id, subdomains)
        if edges:
            self._read_edges()
        else:
            self.edges = False

    def _read_edges(self):
        """reads meshes assuming `mesh` was a xdmf file and edge meshes
        are present in the same directory"""
        path = os.path.dirname(os.path.abspath(self.xdmf_file))
        base = os.path.splitext(os.path.basename(self.xdmf_file))[0]
        ext = os.path.splitext(os.path.basename(self.xdmf_file))[1]

        def read(xdmf):
            mesh = df.Mesh()
            with df.XDMFFile(xdmf) as f:
                f.read(mesh)
            return mesh

        edge_meshes = []
        boundary = ["bottom", "right", "top", "left"]
        for b in boundary:
            edge = path + "/" + base + f"_{b}" + ext
            mesh = read(edge)
            edge_meshes.append(mesh)
        self.edges = tuple(edge_meshes)

    def get_nodes(self, n=4):
        """get nodes of the rectangular domain

        Parameters
        ----------
        n : int, optional
            Number of nodes.

        """

        def midpoint(a, b):
            return a + (b - a) / 2

        nodes = np.array(
            [
                [self.xmin, self.ymin],
                [self.xmax, self.ymin],
                [self.xmax, self.ymax],
                [self.xmin, self.ymax],
                [midpoint(self.xmin, self.xmax), self.ymin],
                [self.xmax, midpoint(self.ymin, self.ymax)],
                [midpoint(self.xmin, self.xmax), self.ymax],
                [self.xmin, midpoint(self.ymin, self.ymax)],
                [midpoint(self.xmin, self.xmax), midpoint(self.ymin, self.ymax)],
            ]
        )
        return nodes[:n]

    def translate(self, point):
        """translate the domain in space

        Parameters
        ----------
        point : dolfin.Point
            The point by which to translate.

        Note: if `self.edges` evaluates to True, edge
        meshes are translated as well.
        """
        self.mesh.translate(point)
        # update edges if True
        if self.edges:
            for edge in self.edges:
                edge.translate(point)


class StructuredGrid(object):
    """class representing a structured (coarse scale) grid

    Each coarse cell is associated with a fine scale grid which
    needs to be set through `self.fine_grids`.
    """

    def __init__(self, points, cells, tdim):
        # FIXME rather read mesh using meshio
        self.points = points
        self.cells = cells
        self.tdim = tdim

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

    # TODO idea
    # define cell sets needed for basis construction
    # using simple functions within_range and plane_at
    # to get the points and then get_cells_by_points to get the cells

    def within_range(self):
        # TODO
        pass

    def plane_at(self):
        # TODO
        pass

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
