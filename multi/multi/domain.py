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

    def __init__(self, mesh, cell_markers=None, facet_markers=None, index=None, edges=False):
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
            boundary_vertices = sorted(dolfinx.mesh.exterior_facet_indices(submesh.topology))

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

    # FIXME this is not predictable
    # def get_cells_points(self, x):
    #     """return all cells containing points given by coordinates `x`"""
    #     try:
    #         x = x.reshape(int(x.size/3), 3)
    #     except ValueError as err:
    #         raise err("x.shape = (num_points, 3) is required!")

    #     # Find cells whose bounding-box collide with the the points
    #     bb_tree = self.bb_tree
    #     cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, x)
    #     assert cell_candidates.num_nodes < 2
    #     return cell_candidates.links(0)

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
