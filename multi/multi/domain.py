import pathlib
import gmsh
import tempfile
import meshio
import numpy as np
import dolfinx
from mpi4py import MPI
from multi.preprocessing import create_mesh


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
        using `dolfinx.mesh.create_submesh`.
        The submesh and associated mappings are stored in a
        dictionary (self.edges).
    """

    def __init__(
        self, mesh, cell_markers=None, facet_markers=None, index=None, edges=False
    ):
        super().__init__(mesh, cell_markers, facet_markers, index)
        if edges:
            self._create_edge_meshes()
        else:
            self.edges = False

    def _create_edge_meshes(self):
        parent = self.mesh
        tdim = parent.topology.dim
        fdim = tdim - 1
        parent.topology.create_connectivity(fdim, tdim)

        xmin, ymin, zmin = self.xmin
        xmax, ymax, zmax = self.xmax

        def bottom(x):
            return np.isclose(x[1], ymin)

        def right(x):
            return np.isclose(x[0], xmax)

        def top(x):
            return np.isclose(x[1], ymax)

        def left(x):
            return np.isclose(x[0], xmin)

        edges = {}
        markers = {"bottom": bottom, "right": right, "top": top, "left": left}
        for key, marker in markers.items():
            facets = dolfinx.mesh.locate_entities_boundary(parent, fdim, marker)
            edges[key] = dolfinx.mesh.create_submesh(parent, fdim, facets)
        self.edges = edges

    # FIXME: remove this?
    # working with the StructuredQuadGrid or DofMap, i.e. the actual
    # coarse grid is much easier to achieve the same thing
    # however, this might be useful if coarse grid is not available
    def get_corner_vertices(self):
        """determine the vertices of the RceDomain

        Returns
        -------
        verts : list of int
            The vertices following local ordering of a
            quadrilateral cell as in multi.dofmap.CellDofLayout.

        """

        def determine_candidates(submesh, parent, parent_facets):
            # need to create connectivity to compute facets
            tdim = submesh.topology.dim
            fdim = tdim - 1
            submesh.topology.create_connectivity(fdim, tdim)
            boundary_vertices = sorted(
                dolfinx.mesh.exterior_facet_indices(submesh.topology)
            )

            child_facets = []
            vertex_to_edge = submesh.topology.connectivity(0, 1)
            for vertex in boundary_vertices:
                child_facets.append(vertex_to_edge.links(vertex))
            child_facets = np.hstack(child_facets)

            parent_facets = np.array(parent_facets)[child_facets]
            parent.topology.create_connectivity(1, 0)
            facet_to_vertex = parent.topology.connectivity(1, 0)
            vertex_candidates = []
            for facet in parent_facets:
                verts = facet_to_vertex.links(facet)
                vertex_candidates.append(verts)
            vertex_candidates = np.hstack(vertex_candidates)

            return vertex_candidates

        parent = self.mesh
        candidates = {}
        for key, stuff in self.edges.items():
            submesh = stuff[0]
            parent_facets = stuff[1]
            candidates[key] = set(determine_candidates(submesh, parent, parent_facets))

        # if this order does not follow multi.dofmap.QuadrilateralDofLayout
        # the ordering of coarse scale basis is incorrect
        v0 = candidates["bottom"].intersection(candidates["left"])
        v1 = candidates["bottom"].intersection(candidates["right"])
        v2 = candidates["left"].intersection(candidates["top"])
        v3 = candidates["right"].intersection(candidates["top"])
        verts = [v0, v1, v2, v3]
        assert all([len(s) == 1 for s in verts])
        return [s.pop() for s in verts]

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
            for edge in self.edges.values():
                domain = edge[0]
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
        self.num_cells = self.mesh.topology.connectivity(2, 0).num_nodes
        self.cells = np.arange(self.num_cells)
        self.tdim = mesh.topology.dim

    @property
    def cell_sets(self):
        return self._cell_sets

    @cell_sets.setter
    def cell_sets(self, pairs):
        """set cell sets for given pairs of key and array of cell indices"""
        self._cell_sets = pairs

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

    def locate_entities(self, dim, marker):
        return dolfinx.mesh.locate_entities(self.mesh, dim, marker)

    def locate_entities_boundary(self, dim, marker):
        assert dim < self.mesh.topology.dim
        return dolfinx.mesh.locate_entities_boundary(self.mesh, dim, marker)

    @property
    def fine_grid_method(self):
        return self._fine_grid_method

    @fine_grid_method.setter
    def fine_grid_method(self, method):
        """set method to create fine grid for coarse grid cell"""
        self._fine_grid_method = method

    def create_fine_grid(self, cells, output, cell_type="triangle", **kwargs):
        """creates a fine scale grid for given cells

        Parameters
        ----------
        cells : np.ndarray
            The cell indices for which to create a fine scale grid.
            Requires `self.fine_grid_method` to be defined.
        output : str
            The path to write the result (suffix .msh).
        cell_type : optional
            The `meshio` cell type of the fine grid.
            Currently, only meshes with one cell type are supported.
        kwargs : optional
            Keyword arguments to be passed to `self.fine_grid_method`.
        """
        # cases: (a) single cell, (b) patch of cells, (c) entire coarse grid

        tdim = self.tdim
        num_cells = kwargs.get("num_cells", 10)

        # initialize
        subdomains = []

        cells = np.array(cells)
        active_cells = self.cells[cells]

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
                create_fine_grid = self._fine_grid_method
                create_fine_grid(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    num_cells=num_cells,
                    facets=False,
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

        gmsh.model.mesh.generate(self.tdim)

        gmsh.write(output)
        gmsh.finalize()

        # convert to xdmf using meshio
        in_mesh = meshio.read(output)
        if tdim < 3:
            prune_z = True
        out_mesh = create_mesh(in_mesh, cell_type, prune_z=prune_z)

        with tempfile.NamedTemporaryFile(suffix=".xdmf") as tf:
            meshio.write(tf.name, out_mesh)

            with dolfinx.io.XDMFFile(MPI.COMM_WORLD, tf.name, "r") as xdmf:
                mesh = xdmf.read_mesh(name="Grid")
                ct = xdmf.read_meshtags(mesh, name="Grid")

            # remove the h5 as well
            h5 = pathlib.Path(tf.name).with_suffix(".h5")
            h5.unlink()

        # clean up
        for msh_file in subdomains:
            pathlib.Path(msh_file).unlink()

        return (mesh, ct)
