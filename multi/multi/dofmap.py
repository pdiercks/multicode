import numpy as np
import basix


class QuadrilateralDofLayout(object):
    """
    NOTE
    this is mainly tested and implemented to be used to
    construct a custom DofMap on quadrilateral meshes

    quadrilateral reference cell topology

    FIXME
    currently DofMap relies on domain.topology to distribute the dofs
    therefore, the `dofs_per_edge` need to be given in the local ordering
    compliant with domain.topology
    Ideally, this local ordering of the dofs should follow the basix element dof layout.

    Order (1)
    ---------
    v2--e3--v3
    |       |
    e1  f0  e2
    |       |
    v0--e0--v1

    see basix.geometry(basix.CellType.quadrilateral)
    and basix.topology(basix.CellType.quadrilateral)

    BUT for now the following ordering of vertices and edges is assumed:

    Order (2)
    ---------
    v1--e2--v3
    |       |
    e0  f0  e3
    |       |
    v0--e1--v2

    EDIT 12.10.2022
    Apparently the built-in mesh has above ordering (2) and
    a mesh read from msh has ordering (1) ...
    --> decision: go with ordering (1) and only use Gmsh, but
    not the built-in meshes with DofMap, StructuredQuadGrid, etc.

    """

    def __init__(self):
        self.topology = basix.topology(basix.CellType.quadrilateral)
        self.geometry = basix.geometry(basix.CellType.quadrilateral)
        self.num_entities = [len(ents) for ents in self.topology]
        self.local_edge_index_map = {
                "left": 1, "bottom": 0, "top": 3, "right": 2,
                1: "left", 0: "bottom", 3: "top", 2: "right"
                }

    def get_entity_dofs(self):
        return self.__entity_dofs

    def set_entity_dofs(self, ndofs_per_ent):
        """set number of dofs per entity

        Parameters
        ----------
        ndofs_per_ent : tuple of int or np.ndarray
            Number of dofs per entity. For the edges this can be
            a numpy array otherwise an integer value is allowed.
        """
        assert len(ndofs_per_ent) == len(self.num_entities)

        self.__entity_dofs = {0: {}, 1: {}, 2: {}}  # key=entity_dim, value=dict
        counter = 0
        for dim, ndofs in enumerate(ndofs_per_ent):
            num_entities = self.num_entities[dim]
            if isinstance(ndofs, int):
                ndofs = [ndofs for _ in range(num_entities)]
            for entity in range(num_entities):
                self.__entity_dofs[dim][entity] = [
                    counter + dof for dof in range(ndofs[entity])
                ]
                counter += ndofs[entity]


class DofMap:
    """class representing a DofMap of a function space where each entity
    (vertex, edge, face in 2d) is associated with the given number of DoFs,
    when calling `distribute_dofs()`.

    Parameters
    ----------
    grid : multi.domain.StructuredQuadGrid
        The quadrilateral mesh of the computational domain.
    """

    def __init__(self, grid):
        self.grid = grid
        self.dof_layout = QuadrilateralDofLayout()

        # create connectivities
        self.conn = []
        domain = grid.grid
        for dim in range(len(self.dof_layout.num_entities)):
            domain.topology.create_connectivity(2, dim)
            self.conn.append(domain.topology.connectivity(2, dim))
        self.num_cells = self.conn[2].num_nodes

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
        self._dm = {dim: {} for dim in dimension}
        DoF = 0

        num_cells = self.num_cells

        if isinstance(dofs_per_edge, (int, np.integer)):
            dofs_per_edge = np.ones((num_cells, 4), dtype=np.intc) * dofs_per_edge
        else:
            assert dofs_per_edge.shape == (num_cells, 4)

        for cell_index in range(num_cells):
            self.dof_layout.set_entity_dofs(
                (dofs_per_vert, dofs_per_edge[cell_index], dofs_per_face)
            )
            entity_dofs = self.dof_layout.get_entity_dofs()
            for dim, conn in enumerate(self.conn):
                entities = conn.links(cell_index)
                for local_ent, ent in enumerate(entities):
                    if ent not in self._dm[dim].keys():
                        self._dm[dim][ent] = []
                        dofs = entity_dofs[dim][local_ent]
                        for dof in dofs:
                            self._dm[dim][ent].append(DoF)
                            DoF += 1

        self._n_dofs = DoF
        self.dofs_per_vert = dofs_per_vert
        self.dofs_per_edge = dofs_per_edge
        self.dofs_per_face = dofs_per_face

    @property
    def num_dofs(self):
        """total number of dofs"""
        if not hasattr(self, "_n_dofs"):
            raise AttributeError("You need to distribute DoFs first")
        return self._n_dofs

    def cell_dofs(self, cell_index):
        """returns dofs for given cell

        Returns
        -------
        dofs : list of int
            The dofs of the given cell.
        """
        if not hasattr(self, "_dm"):
            raise AttributeError("You need to distribute DoFs first")

        num_cells = self.num_cells
        assert cell_index in np.arange(num_cells)

        cell_dofs = []
        for dim, conn in enumerate(self.conn):
            entities = conn.links(cell_index)
            for ent in entities:
                cell_dofs += self._dm[dim][ent]
        return cell_dofs

    def entity_dofs(self, dim, entity):
        """return all dofs for entity `entity` of dimension `dim`"""
        return self._dm[dim][entity]
