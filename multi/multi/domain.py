import pathlib
import gmsh
import tempfile
import meshio
import numpy as np
import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
from multi.preprocessing import create_mesh, create_line_grid


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


class RceDomain(Domain):
    """
    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The partition of the representative coarse grid element.
    cell_markers : optional
    facet_markers : optional
    index : optional, int
        The identification number of the domain.
    edges : optional, bool
        If True, create meshes for the edges of the domain
        using `multi.preprocessing.create_line_grid`.
        Note that `mesh` needs to have equispaced (transfinite lines)
        nodes on the boundary.
    """

    def __init__(
        self, mesh, cell_markers=None, facet_markers=None, index=None
    ):
        super().__init__(mesh, cell_markers, facet_markers, index)

    def create_edge_meshes(self, num_cells=None):
        parent = self.mesh
        tdim = parent.topology.dim
        fdim = tdim - 1
        parent.topology.create_connectivity(fdim, tdim)
        facets = dolfinx.mesh.locate_entities_boundary(parent, fdim, lambda x: np.full(x[0].shape, True, dtype=bool))
        num_cells = num_cells or int(facets.size / 4)

        xmin, ymin, zmin = self.xmin
        xmax, ymax, zmax = self.xmax

        # FIXME sadly cannot use dolfinx.mesh.create_submesh
        # see test/test_edge_spaces.py

        # def bottom(x):
        #     return np.isclose(x[1], ymin)

        # def right(x):
        #     return np.isclose(x[0], xmax)

        # def top(x):
        #     return np.isclose(x[1], ymax)

        # def left(x):
        #     return np.isclose(x[0], xmin)

        edges = {}
        # markers = {"bottom": bottom, "right": right, "top": top, "left": left}
        # for key, marker in markers.items():
        #     facets = dolfinx.mesh.locate_entities_boundary(parent, fdim, marker)
        #     edges[key] = dolfinx.mesh.create_submesh(parent, fdim, facets)
        points = {
                "bottom": ([xmin, ymin, 0.], [xmax, ymin, 0.]),
                "left": ([xmin, ymin, 0.], [xmin, ymax, 0.]),
                "right": ([xmax, ymin, 0.], [xmax, ymax, 0.]),
                "top": ([xmin, ymax, 0.], [xmax, ymax, 0.])
                }
        for key, (start, end) in points.items():
            with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
                create_line_grid(start, end, num_cells=num_cells, out_file=tf.name)
                domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
            edges[key] = domain
        self.edges = edges

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
        # update child meshes as well
        if self.edges:
            for domain in self.edges.values():
                xg = domain.geometry.x
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
        self.num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        self.cells = np.arange(self.num_cells, dtype=np.int32)
        self.tdim = mesh.topology.dim

    def get_patch(self, cell_index):
        """return all cells neighbouring cell with index `cell_index`"""
        point_tags = self.get_entities(0, cell_index)
        conn_02 = self.mesh.topology.connectivity(0, 2)
        cells = list()
        for tag in point_tags:
            cells.append(conn_02.links(tag))
        return np.unique(np.hstack(cells))

    def get_cells(self, dim, entities):
        """return cells containing entities of dimension `dim`"""
        ent_to_cell = self.mesh.topology.connectivity(dim, 2)
        cells = list()
        for tag in entities.flatten():
            candidates = ent_to_cell.links(tag)
            cells.append(candidates)
        return np.unique(cells)

    def get_entities(self, dim, cell_index):
        """get entities of dimension `dim` for cell with index `cell_index`"""
        assert dim in (0, 1)
        conn = self.mesh.topology.connectivity(2, dim)
        return conn.links(cell_index)

    def get_entity_coordinates(self, dim, entities):
        """return coordinates of `entities` of dimension `dim`"""
        return dolfinx.mesh.compute_midpoints(self.mesh, dim, entities)

    def locate_entities(self, dim, marker):
        """locate entities of `dim` geometrically using `marker`"""
        return dolfinx.mesh.locate_entities(self.mesh, dim, marker)

    def locate_entities_boundary(self, dim, marker):
        """locate entities of `dim` on the boundary geometrically using `marker`"""
        return dolfinx.mesh.locate_entities_boundary(self.mesh, dim, marker)

    @property
    def fine_grid_method(self):
        """methods to create fine grid for each coarse grid cell"""
        return self._fine_grid_method

    @fine_grid_method.setter
    def fine_grid_method(self, methods):
        """set methods to create fine grid for each coarse grid cell"""
        try:
            num = len(methods)
            if not num == self.num_cells:
                raise ValueError
        except TypeError:
            # set same method for every cell
            methods = [
                methods,
            ] * self.num_cells
        self._fine_grid_method = methods

    def create_fine_grid(self, cells, output, cell_type="triangle", **kwargs):
        """creates a fine scale grid for given cells

        Parameters
        ----------
        cells : np.ndarray
            The cell indices for which to create a fine scale grid.
            Requires `self.fine_grid_method` to be defined.
        output : str
            The path to write the result (suffix .xdmf).
        cell_type : optional
            The `meshio` cell type of the fine grid.
            Currently, only meshes with one cell type are supported.
        kwargs : optional
            Keyword arguments to be passed to `self.fine_grid_method`.
        """
        # cases: (a) single cell, (b) patch of cells, (c) entire coarse grid

        # meshio cell types for mesh creation
        assert cell_type in ("triangle", "quad")
        facet_cell_type = "line"

        tdim = self.tdim
        num_cells = kwargs.get("num_cells", 10)

        # initialize
        subdomains = []

        cells = np.array(cells)
        assert cells.size > 0
        active_cells = self.cells[cells]
        create_facets = cells.size < 2

        for cell in active_cells:
            vertices = self.get_entities(0, cell)
            dx = dolfinx.mesh.compute_midpoints(self.mesh, 0, vertices)
            dx = np.around(dx, decimals=3)
            xmin, ymin, zmin = dx[0]
            xmax, ymax, zmax = dx[3]
            assert xmin < xmax
            assert ymin < ymax

            # create msh file using self._fine_grid_method
            with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tf:
                subdomains.append(tf.name)
                create_fine_grid = self.fine_grid_method[cell]
                create_fine_grid(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    num_cells=num_cells,
                    facets=create_facets,
                    out_file=tf.name,
                )

        # merge subdomains
        gmsh.initialize()
        gmsh.clear()
        gmsh.model.add("fine_grid")
        gmsh.option.setNumber("General.Verbosity", 0)  # silent except for fatal errors

        for msh_file in subdomains:
            gmsh.merge(msh_file)

        gmsh.model.geo.remove_all_duplicates()
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.remove_duplicate_nodes()
        gmsh.model.mesh.remove_duplicate_elements()

        gmsh.model.mesh.generate(tdim)

        tf_msh = tempfile.NamedTemporaryFile(suffix=".msh")
        gmsh.write(tf_msh.name)
        gmsh.finalize()

        # convert to xdmf using meshio
        # reasoning: always convert to xdmf since (merged) msh
        # cannot be read by dolfinx.io.gmshio
        in_mesh = meshio.read(tf_msh.name)
        if tdim < 3:
            prune_z = True
        cell_mesh = create_mesh(in_mesh, cell_type, prune_z=prune_z)
        meshio.write(output, cell_mesh)
        if create_facets:
            outfile = pathlib.Path(output)
            facet_output = outfile.parent / (outfile.stem + "_facets.xdmf")
            facet_mesh = create_mesh(in_mesh, facet_cell_type, prune_z=prune_z)
            meshio.write(facet_output.as_posix(), facet_mesh)

        # clean up
        tf_msh.close()
        for msh_file in subdomains:
            pathlib.Path(msh_file).unlink()
