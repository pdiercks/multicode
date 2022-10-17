import os
import gmsh
import tempfile
import meshio
import numpy as np
import dolfinx
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


def extract_to_meshio():
    # extract point coords
    idx, points, _ = gmsh.model.mesh.getNodes()
    points = np.asarray(points).reshape(-1, 3)
    idx -= 1
    srt = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    points = points[srt]

    # extract cells
    elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements()
    cells = []
    for elem_type, elem_tags, node_tags in zip(elem_types, elem_tags, node_tags):
        # `elementName', `dim', `order', `numNodes', `localNodeCoord',
        # `numPrimaryNodes'
        num_nodes_per_cell = gmsh.model.mesh.getElementProperties(elem_type)[3]

        node_tags_reshaped = np.asarray(node_tags).reshape(-1, num_nodes_per_cell) - 1
        node_tags_sorted = node_tags_reshaped[np.argsort(elem_tags)]
        cells.append(
            meshio.CellBlock(
                meshio.gmsh.gmsh_to_meshio_type[elem_type], node_tags_sorted
            )
        )

    cell_sets = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        cell_sets[name] = [[] for _ in range(len(cells))]
        for e in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            # TODO node_tags?
            # elem_types, elem_tags, node_tags
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim, e)
            assert len(elem_types) == len(elem_tags)
            assert len(elem_types) == 1
            elem_type = elem_types[0]
            elem_tags = elem_tags[0]

            meshio_cell_type = meshio.gmsh.gmsh_to_meshio_type[elem_type]
            # make sure that the cell type appears only once in the cell list
            # -- for now
            idx = []
            for k, cell_block in enumerate(cells):
                if cell_block.type == meshio_cell_type:
                    idx.append(k)
            assert len(idx) == 1
            idx = idx[0]
            cell_sets[name][idx].append(elem_tags - 1)

        cell_sets[name] = [
            (None if len(idcs) == 0 else np.concatenate(idcs))
            for idcs in cell_sets[name]
        ]

    # make meshio mesh
    # TODO add cell data ....
    return meshio.Mesh(points, cells, cell_sets=cell_sets)


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
    def fine_grids(self):
        return self._fine_grids

    @fine_grids.setter
    def fine_grids(self, values):
        """values as array of length (num_cells,) holding path to fine grid"""
        # TODO only support .msh format for fine grids of this class
        self._fine_grids = values

    def create_fine_grid(self, cells, output, cell_type="triangle"):
        """creates a fine scale grid for given cells

        Parameters
        ----------
        cells : np.ndarray
            The cell indices for which to create a fine scale grid.
            Requires `self.fine_grids` to be defined.
        output : str
            The path to write the result (suffix .msh).
        cell_type : optional
            The `meshio` cell type of the fine grid.
        """
        # cases: (a) single cell, (b) patch of cells, (c) entire coarse grid

        tdim = self.tdim

        # initialize
        subdomains = []

        cells = np.array(cells)
        active_cells = self.cells[cells]
        fine_grids = self.fine_grids[cells]

        for cell, grid_path in zip(active_cells, fine_grids):
            vertices = self.get_entities(0, cell)
            dx = dolfinx.mesh.compute_midpoints(self.mesh, 0, vertices[0])
            dx = np.around(dx, decimals=3)

            try:
                instream = grid_path.as_posix()
            except AttributeError:
                instream = grid_path
            mesh = meshio.read(instream)
            out_mesh = create_mesh(mesh, cell_type)

            # translation
            out_mesh.points += dx

            assert "gmsh:physical" in out_mesh.cell_data.keys()
            # true for rce_05.msh, test case
            cell_data = out_mesh.get_cell_data("gmsh:physical", "triangle")
            assert cell_data.shape[0] == 160
            assert np.isclose(np.sum(cell_data), 192)
            # inclusion: 32 cells, matrix: 128 cells; 128 * 1 + 32 * 2 = 192

            # FIXME this is not working at all as expected
            with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tf:
                subdomains.append(tf.name)
                meshio.write(tf.name, out_mesh, file_format="gmsh22", binary=False)
                # meshio.write(tf.name, out_mesh, file_format="gmsh")

        # merge subdomains
        gmsh.initialize()
        gmsh.clear()
        gmsh.model.add("fine_grid")

        for msh_file in subdomains:
            gmsh.merge(msh_file)
        assert len(subdomains) == 1

        gmsh.model.geo.remove_all_duplicates()
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.remove_duplicate_nodes()
        gmsh.model.mesh.remove_duplicate_elements()

        gmsh.model.mesh.generate(self.tdim)

        gmsh.write(output)
        gmsh.finalize()


        # test = meshio.read(output)
        breakpoint()
        cd = test.get_cell_data("gmsh:physical", "triangle")
        assert np.sum(cd) > 160

        # FIXME Write final mesh to xdmf instead of msh?
        # currently the created msh cannot be imported with gmshio.read_from_msh

        # clean up
        for msh_file in subdomains:
            os.remove(msh_file)
