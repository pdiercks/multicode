import pathlib
import gmsh
import tempfile
import meshio
import numpy as np
import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
from multi.preprocessing import create_mesh, create_line_grid, create_rectangle_grid


class Domain(object):
    """Class to represent a computational domain

    Parameters
    ----------
    grid : dolfinx.mesh.Mesh
        The partition of the domain.
    cell_markers : TODO
    facet_markers : TODO
    index : optional, int
        The identification number of the domain.
    """

    def __init__(self, grid, cell_markers=None, facet_markers=None, index=None):
        self.grid = grid
        self.cell_markers = cell_markers
        self.facet_markers = facet_markers
        self.index = index
        self._x = grid.geometry.x

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
    """representation of a rectangular domain Ω=[xs, xe]x[ys, ye]

    Parameters
    ----------
    grid : dolfinx.mesh.Mesh
        The fine grid partition of the domain.
    cell_markers : optional, dolfinx.mesh.MeshTags
        Mesh tags for the cells of the domain.
    facet_markers : optional, dolfinx.mesh.MeshTags
        Mesh tags for the facets of the domain.
    index : optional, int
        The identification number of the domain.
    """

    def __init__(
        self, grid, cell_markers=None, facet_markers=None, index=None
    ):
        super().__init__(grid, cell_markers, facet_markers, index)

    def create_coarse_grid(self):
        """create a coarse grid partition of Ω"""

        xmin, ymin, zmin = self.xmin
        xmax, ymax, zmax = self.xmax

        with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
            create_rectangle_grid(
                    xmin, xmax, ymin, ymax, 0.,
                    recombine=True, num_cells=1, out_file=tf.name)
            coarse, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
        self.coarse_grid = coarse

    def create_edge_meshes(self, num_cells=None):
        """create coarse and fine grid partitions of the boundary of Ω"""
        parent = self.grid
        tdim = parent.topology.dim
        fdim = tdim - 1
        parent.topology.create_connectivity(fdim, tdim)
        facets = dolfinx.mesh.locate_entities_boundary(parent, fdim, lambda x: np.full(x[0].shape, True, dtype=bool))
        # assumes a quadrilateral domain and equal number of facets
        # per boundary/edge
        num_cells = num_cells or int(facets.size / 4)

        xmin, ymin, zmin = self.xmin
        xmax, ymax, zmax = self.xmax

        fine_grid = {}
        coarse_grid = {}

        points = {
                "bottom": ([xmin, ymin, 0.], [xmax, ymin, 0.]),
                "left": ([xmin, ymin, 0.], [xmin, ymax, 0.]),
                "right": ([xmax, ymin, 0.], [xmax, ymax, 0.]),
                "top": ([xmin, ymax, 0.], [xmax, ymax, 0.])
                }
        for key, (start, end) in points.items():
            with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
                create_line_grid(start, end, num_cells=num_cells, out_file=tf.name)
                fine, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
            fine_grid[key] = fine
            with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
                create_line_grid(start, end, num_cells=1, out_file=tf.name)
                coarse, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
            coarse_grid[key] = coarse

        self.fine_edge_grid = fine_grid
        self.coarse_edge_grid = coarse_grid



class StructuredQuadGrid(object):
    """class representing a structured (coarse scale) quadrilateral grid

    Each coarse quadrilateral cell is associated with a fine scale grid which
    can be created by setting the property `self.fine_grid_method` and
    calling `self.create_fine_grid`.

    Parameters
    ----------
    grid : dolfinx.mesh.Mesh
        The coarse grid partition of the domain.
    cell_markers : TODO
    facet_markers : TODO
    """

    def __init__(self, grid, cell_markers=None, facet_markers=None):
        self.grid = grid
        self.cell_markers = cell_markers
        self.facet_markers = facet_markers

        # bounding box tree
        self.bb_tree = dolfinx.geometry.BoundingBoxTree(grid, grid.topology.dim)

        grid.topology.create_connectivity(2, 0)
        grid.topology.create_connectivity(2, 1)
        grid.topology.create_connectivity(0, 2)
        self.num_cells = grid.topology.index_map(grid.topology.dim).size_local
        self.cells = np.arange(self.num_cells, dtype=np.int32)
        self.tdim = grid.topology.dim

    def get_patch(self, cell_index):
        """return all cells neighbouring cell with index `cell_index`"""
        point_tags = self.get_entities(0, cell_index)
        conn_02 = self.grid.topology.connectivity(0, 2)
        cells = list()
        for tag in point_tags:
            cells.append(conn_02.links(tag))
        return np.unique(np.hstack(cells))

    def get_cells(self, dim, entities):
        """return cells containing entities of dimension `dim`"""
        ent_to_cell = self.grid.topology.connectivity(dim, 2)
        cells = list()
        for tag in entities.flatten():
            candidates = ent_to_cell.links(tag)
            cells.append(candidates)
        return np.unique(cells)

    def get_entities(self, dim, cell_index):
        """get entities of dimension `dim` for cell with index `cell_index`"""
        assert dim in (0, 1)
        conn = self.grid.topology.connectivity(2, dim)
        return conn.links(cell_index)

    def get_entity_coordinates(self, dim, entities):
        """return coordinates of `entities` of dimension `dim`"""
        return dolfinx.mesh.compute_midpoints(self.grid, dim, entities)

    def locate_entities(self, dim, marker):
        """locate entities of `dim` geometrically using `marker`"""
        return dolfinx.mesh.locate_entities(self.grid, dim, marker)

    def locate_entities_boundary(self, dim, marker):
        """locate entities of `dim` on the boundary geometrically using `marker`"""
        return dolfinx.mesh.locate_entities_boundary(self.grid, dim, marker)

    @property
    def fine_grid_method(self):
        """methods to create fine grid for each coarse grid cell"""
        return self._fine_grid_method

    @fine_grid_method.setter
    def fine_grid_method(self, methods):
        """set methods to create fine grid for each coarse grid cell

        Parameters
        ----------
        methods : list of str or list of callable
            The method can either be a function with an appropriate
            signature (see e.g. multi.preprocessing.create_rectangle_grid)
            or a `str`, i.e. the filepath to a reference mesh
            that should be duplicated for the respective coarse grid cell.
            If the length of list is 1, the same 'method' is used for 
            each coarse grid cell.
        """
        if len(methods) < 2:
            methods *= self.num_cells
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
        num_cells = kwargs.get("num_cells")

        # initialize
        subdomains = []

        cells = np.array(cells)
        assert cells.size > 0
        active_cells = self.cells[cells]
        if active_cells.size > 1:
            create_facets = False
        else:
            create_facets = True

        for cell in active_cells:
            vertices = self.get_entities(0, cell)
            dx = dolfinx.mesh.compute_midpoints(self.grid, 0, vertices)
            dx = np.around(dx, decimals=3)
            xmin, ymin, zmin = dx[0]
            xmax, ymax, zmax = dx[3]
            assert xmin < xmax
            assert ymin < ymax

            fine_grid_method = self.fine_grid_method[cell]

            # ### Subdomain instantiation
            with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tf:
                subdomains.append(tf.name)

                if isinstance(fine_grid_method, str):
                    # read msh file and translate, then save to msh again
                    subdomain_mesh = meshio.read(fine_grid_method)
                    subdomain_mesh.points += dx[0]
                    meshio.write(tf.name, subdomain_mesh, file_format="gmsh")
                else:
                    # create msh via method
                    fine_grid_method(
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
        gmsh.model.add("fine_grid")
        gmsh.option.setNumber("General.Verbosity", 0)  # silent except for fatal errors
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Geometry.AutoCoherence", 2)

        for msh_file in subdomains:
            gmsh.merge(msh_file)

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
