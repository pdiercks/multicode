import warnings
from typing import Optional, Callable
from mpi4py import MPI
import pathlib
import gmsh
import tempfile
import meshio
import numpy as np
import numpy.typing as npt
from dolfinx import mesh, geometry
from dolfinx.io import gmshio
from multi.boundary import plane_at
from multi.preprocessing import create_mesh, create_rectangle, create_meshtags


class Domain(object):
    """Class to represent a computational domain."""

    def __init__(self, grid: mesh.Mesh, cell_tags: Optional[mesh.MeshTags] = None, facet_tags: Optional[mesh.MeshTags] = None):
        """Initializes a domain object.

        Args:
            grid: The partition of the computational domain.
            cell_tags: Mesh tags for cells.
            facet_tags: Mesh tags for facets.

        """
        self.grid = grid
        if cell_tags is not None:
            entities = cell_tags.find(0)
            if entities.size > 0:
                raise ValueError("Cell tags should start at 1."
                                 "Found {entities.size} entities marked with 0.")
        if facet_tags is not None:
            entities = facet_tags.find(0)
            if entities.size > 0:
                raise ValueError("Facet tags should start at 1."
                                 "Found {entities.size} entities marked with 0.")

        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self._x = grid.geometry.x
        self.tdim = grid.topology.dim
        self.gdim = grid.geometry.dim

    @property
    def num_cells(self):
        """Returns number of cells owned by this process"""
        cmap = self.grid.topology.index_map(self.tdim)
        num_cells = cmap.size_local
        return num_cells

    def translate(self, dx):
        """Translates the domain by `dx`."""
        dx = np.array(dx)
        self._x += dx

    @property
    def xmin(self):
        return np.amin(self._x, axis=0)

    @property
    def xmax(self):
        return np.amax(self._x, axis=0)


class RectangularDomain(Domain):
    """Discretization of a rectangular domain Ω=[xs, xe]x[ys, ye]."""
    boundaries = ["bottom", "left", "right", "top"]

    def __init__(self, grid: mesh.Mesh, cell_tags: Optional[mesh.MeshTags] = None, facet_tags: Optional[mesh.MeshTags] =  None):
        """Initializes a domain object.

        Args:
            grid: The partition of the domain.
            cell_tags: Mesh tags of the cells.
            facet_tags: Mesh tags for the facets.
            
        """
        super().__init__(grid, cell_tags, facet_tags)

    def _init_markers(self) -> dict[str, Callable]:
        xmin, ymin, _ = self.xmin
        xmax, ymax, _ = self.xmax
        return {
            "left" : plane_at(xmin, "x"),
            "right" : plane_at(xmax, "x"),
            "bottom" : plane_at(ymin, "y"),
            "top" : plane_at(ymax, "y"),
            }

    def str_to_marker(self, boundary: str) -> Callable:
        """Returns a marker function for `boundary`.

        Args:
            boundary: The boundary of the rectangular domain.
        """
        supported = set(self.boundaries)
        markers = self._init_markers()
        if boundary not in supported:
            raise ValueError(f"{boundary=} does not match. Supported values are {supported}.")
        return markers[boundary]

    def create_facet_tags(self, boundaries: dict[str, int]) -> None:
        """Creates facet tags for given boundaries.

        Args:
            boundaries: A dict mapping boundary to meshtag.

        Note:
            See `self.boundaries` for appropriate values for dict keys.

        """
        if self.facet_tags is not None:
            warnings.warn("Facet tags already exist for this domain and will be overridden!")

        fmarkers = {}
        supported = set(self.boundaries)
        _markers = self._init_markers()
        for boundary, tag in boundaries.items():
            if boundary not in supported:
                raise ValueError(f"{boundary=} does not match. Supported values are {supported}.")
            marker = _markers[boundary]
            fmarkers[boundary] = (tag, marker)

        facet_tags, _ = create_meshtags(self.grid, self.tdim-1, fmarkers)
        self.facet_tags = facet_tags


class RectangularSubdomain(RectangularDomain):
    """A subdomain in a multiscale context."""

    def __init__(self, id: int, grid: mesh.Mesh, cell_tags: Optional[mesh.MeshTags] = None, facet_tags: Optional[mesh.MeshTags] = None):
        """Initializes a subdomain object.

        Args:
            id: Identification number of the subdomain.
            grid: The (fine) grid partition of the subdomain.
            cell_tags: Mesh tags for the cells.
            facet_tags: Mesh tags for the facets.

        """
        super().__init__(grid, cell_tags, facet_tags)
        self.id = id

    def create_coarse_grid(self, num_cells: int, recombine: bool = True) -> None:
        """Creates a coarse grid partition of the subdomain.

        Args:
            num_cells: Number of cells in each spatial direction.
            recombine: If True, recombine triangles into quadrilaterals.

        """

        if hasattr(self, "coarse_grid"):
            raise AttributeError("Coarse grid already exists")

        xmin, ymin, _ = self.xmin
        xmax, ymax, _ = self.xmax
        cmap = self.grid.geometry.cmap
        geom_deg = cmap.degree

        with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
            create_rectangle(
                    xmin, xmax, ymin, ymax, 0.,
                    recombine=recombine, num_cells=num_cells, out_file=tf.name,
                    options={"Mesh.ElementOrder": geom_deg})
            coarse, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_SELF, gdim=self.gdim)
        self.coarse_grid = coarse

    def create_boundary_grids(self) -> None:
        """Creates coarse and fine grid partitions of the boundary of the rectangular subdomain."""
        _markers = self._init_markers()

        if not hasattr(self, "coarse_grid"):
            raise AttributeError("Coarse grid discretization does not exist.")

        def create_boundary_grid(parent, boundary):
            tdim = parent.topology.dim
            fdim = tdim - 1
            parent.topology.create_connectivity(fdim, tdim)
            boundary_marker = _markers[boundary]
            facets = mesh.locate_entities_boundary(parent, fdim, boundary_marker)
            submesh, _, _, _ = mesh.create_submesh(parent, fdim, facets)
            return submesh

        fine_edge_grids = {}
        coarse_edge_grids = {}

        for edge in self.boundaries:
            fine_edge_grids[edge] = create_boundary_grid(self.grid, edge)
            coarse_edge_grids[edge] = create_boundary_grid(self.coarse_grid, edge)

        self.fine_edge_grid = fine_edge_grids
        self.coarse_edge_grid = coarse_edge_grids


class StructuredQuadGrid(object):
    """Class representing a structured (coarse scale) quadrilateral grid.

    Each coarse quadrilateral cell is associated with a fine scale grid which
    can be created by setting the property `self.fine_grid_method` and
    calling `self.create_fine_grid`.

    """

    def __init__(self, grid: mesh.Mesh, cell_tags: Optional[mesh.MeshTags] = None, facet_tags: Optional[mesh.MeshTags] = None):
        """Initializes the grid.

        Args:
            grid: The coarse grid partition of the domain.
            cell_tags: Mesh tags for cells.
            facet_tags: Mesh tags for facets.

        """
        if not grid.topology.dim == 2:
            raise NotImplementedError("Only grids of tdim=2 are supported!")

        cell_type = grid.basix_cell()
        if not cell_type.name == "quadrilateral":
            raise ValueError(f"Expected cell type 'quadrilateral'. Got {cell_type=}")

        self.grid = grid
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags

        # bounding box tree
        self.bb_tree = geometry.bb_tree(grid, grid.topology.dim)

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
        return mesh.compute_midpoints(self.grid, dim, entities)

    def locate_entities(self, dim, marker):
        """locate entities of `dim` geometrically using `marker`"""
        return mesh.locate_entities(self.grid, dim, marker)

    def locate_entities_boundary(self, dim, marker):
        """locate entities of `dim` on the boundary geometrically using `marker`"""
        return mesh.locate_entities_boundary(self.grid, dim, marker)

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
            signature (see e.g. multi.preprocessing.create_rectangle)
            or a `str`, i.e. the filepath to a reference mesh
            that should be duplicated for the respective coarse grid cell.
            If the length of list is 1, the same 'method' is used for 
            each coarse grid cell.
        """
        if len(methods) < 2:
            methods *= self.num_cells
        self._fine_grid_method = methods

    def create_fine_grid(self, cells: npt.NDArray, output: str, cell_type: str, **kwargs) -> None:
        """Creates a fine scale grid for given cells.

        Args:
            cells: Active coarse grid cells for which to create fine scale grid.
            output: The filepath to write the result to.
            cell_type: The meshio cell type of the cells.
            kwargs: Optional keyword arguments to be passed to `self.fine_grid_method`.
        """

        # meshio cell types for mesh creation
        if cell_type in ("triangle", "quad"):
            facet_cell_type = "line"
        elif cell_type in ("triangle6", "quad9"):
            facet_cell_type = "line3"
        else:
            raise NotImplementedError

        def cell_type_mismatch(mesh, cell_type) -> bool:
            """Returns True if there is a mismatch"""
            # FIXME
            # meshes with more than one cell type are not supported
            blocks = mesh.cells
            types = [cb.type for cb in blocks]
            return not all([t == cell_type for t in types])

        # initialize
        tdim = self.tdim
        subdomains = []

        # cases: (a) single cell, (b) patch of cells, (c) entire coarse grid
        cells = np.array(cells)
        assert cells.size > 0
        active_cells = self.cells[cells]
        # if active_cells.size > 1:
        #     create_facets = False
        # else:
        #     create_facets = True

        for cell in active_cells:
            vertices = self.get_entities(0, cell)
            dx = mesh.compute_midpoints(self.grid, 0, vertices)
            dx = np.around(dx, decimals=3)

            xmin = np.amin(dx, axis=0, keepdims=True)
            xmax = np.amax(dx, axis=0, keepdims=True)

            fine_grid_method = self.fine_grid_method[cell]

            # ### Subdomain instantiation
            with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tf:
                subdomains.append(tf.name)

                if isinstance(fine_grid_method, str):
                    # duplicate given mesh for current coarse grid cell
                    # read msh file and translate, then save to msh again
                    subdomain_mesh = meshio.read(fine_grid_method)
                    if cell_type_mismatch(subdomain_mesh, cell_type):
                        raise ValueError(f"Cell type of input mesh and given cell type ({cell_type}) do not match!")
                    subdomain_gdim = subdomain_mesh.points.shape[1]
                    subdomain_mesh.points += xmin[:, :subdomain_gdim]
                    meshio.write(tf.name, subdomain_mesh, file_format="gmsh")
                else:
                    # create grid via method for current coarse grid cell
                    # FIXME: hardcoded interface of `fine_grid_method`
                    fine_grid_method(
                        xmin[:, 0].item(),
                        xmax[:, 0].item(),
                        xmin[:, 1].item(),
                        xmax[:, 1].item(),
                        out_file=tf.name,
                        **kwargs,
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
        prune_z = True
        if 'gmsh:physical' in in_mesh.cell_data_dict.keys():
            name_to_read = 'gmsh:physical'
        else:
            name_to_read = None

        cell_mesh = create_mesh(in_mesh, cell_type, prune_z=prune_z, name_to_read=name_to_read)
        meshio.write(output, cell_mesh)
        # if create_facets:
        #     outfile = pathlib.Path(output)
        #     facet_output = outfile.parent / (outfile.stem + "_facets.xdmf")
        #     facet_mesh = create_mesh(in_mesh, facet_cell_type, prune_z=prune_z)
        #     meshio.write(facet_output.as_posix(), facet_mesh)

        # clean up
        tf_msh.close()
        for msh_file in subdomains:
            pathlib.Path(msh_file).unlink()
