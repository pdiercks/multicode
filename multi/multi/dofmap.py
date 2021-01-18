import logging
from pathlib import Path
import numpy as np
import meshio
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


def adjacency_graph(cells, cell_type="quad"):
    N = np.unique(cells).size  # number of points
    r = np.zeros((N, N), dtype=int)

    if cell_type == "quad8":
        local_adj = {
            0: (4, 7),
            1: (4, 5),
            2: (5, 6),
            3: (6, 7),
            4: (0, 1),
            5: (1, 2),
            6: (2, 3),
            7: (0, 3),
        }
    elif cell_type == "quad":
        local_adj = {
            0: (1, 3),
            1: (2, 0),
            2: (3, 1),
            3: (0, 2),
        }
    elif cell_type == "line3":
        local_adj = {
            0: (2,),
            1: (2,),
            2: (1, 0),
        }
    else:
        raise NotImplementedError

    for cell in cells:
        for vertex, neighbours in local_adj.items():
            for n in neighbours:
                r[cell[vertex], cell[n]] = 1
    return r


class Quadrilateral:
    """
    v3--e2--v2
    |       |
    e3      e1
    |       |
    v0--e0--v1
    """

    def __init__(self):
        self.verts = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        self.edges = {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 1)}
        self.faces = {0: (0, 1, 2, 3)}
        self.topology = {
            0: {0: (0,), 1: (1,), 2: (2,), 3: (3,)},
            1: self.edges,
            2: self.faces,
        }

    def set_entity_dofs(self, dofs_per_vert, dofs_per_edge, dofs_per_face):
        n_vertex_dofs = len(self.verts) * dofs_per_vert
        n_edge_dofs = len(self.edges) * dofs_per_edge
        #  n_face_dofs = len(self.faces) * dofs_per_face
        self.entity_dofs = {0: {}, 1: {}, 2: {}}
        for v in range(len(self.verts)):
            self.entity_dofs[0][v] = [
                dofs_per_vert * v + i for i in range(dofs_per_vert)
            ]
        for e in range(len(self.edges)):
            self.entity_dofs[1][e] = [
                n_vertex_dofs + dofs_per_edge * e + i for i in range(dofs_per_edge)
            ]
        for f in range(len(self.faces)):
            self.entity_dofs[2][f] = [
                n_edge_dofs + dofs_per_face * f + i for i in range(dofs_per_face)
            ]


class DofMap:
    """class representing a DofMap of a function space where each entity
    (vertex, edge, face in 2d) is associated with the given number of DoFs,
    when calling `distribute_dofs()`.

    Parameters
    ----------
    mshfile
        Filepath (incl. ext) or instance of `meshio._mesh.Mesh`.
    tdim : int
        Topological dimension of the mesh.
    gdim : int
        Geometrical dimension of the mesh.
    prune_z_0 : bool, optional
        If True, prune zero z component.
    cell : optional
        Reference cell type.
    rcm : bool, optional
        Reorder points using the Reverse Cuthill McKee algorithm.

    """

    def __init__(
        self, mshfile, tdim, gdim, prune_z_0=True, cell=Quadrilateral, rcm=False
    ):
        if not meshio.__version__ == "4.3.1":
            raise NotImplementedError
        logger = logging.getLogger("DofMap")

        if isinstance(mshfile, (str, Path)):
            logger.info("Reading Gmsh mesh into meshio from path {}".format(mshfile))
            mesh = meshio.read(mshfile)
        elif isinstance(mshfile, meshio._mesh.Mesh):
            mesh = mshfile
        else:
            raise TypeError

        if prune_z_0:
            mesh.prune_z_0()

        # set active coordinate components
        points_pruned = mesh.points[:, :gdim]

        cell_types = {  # meshio cell types per topological dimension
            3: ["tetra", "hexahedron", "tetra10", "hexahedron20"],
            2: ["triangle", "quad", "triangle6", "quad8"],
            1: ["line", "line3"],
            0: ["vertex"],
        }

        # Extract relevant cell blocks depending on supported cell types
        subdomains_celltypes = list(
            set([cb.type for cb in mesh.cells if cb.type in cell_types[tdim]])
        )
        assert len(subdomains_celltypes) <= 1

        subdomains_celltype = (
            subdomains_celltypes[0] if len(subdomains_celltypes) > 0 else None
        )
        if not subdomains_celltype == "quad8":
            # TODO
            # 1. definition of other cell types
            # 2. instantiate cell with mesh data; such that `distribute_dofs` can be generalized
            # e.g. something like
            # Cell = Quadrilateral(self.cells[0])
            # entities = Cell.entities(dim)
            raise NotImplementedError("Currently only cell type 'quad8' is supported.")

        if subdomains_celltype is not None:
            subdomains_cells = mesh.get_cells_type(subdomains_celltype)
        else:
            subdomains_cells = []

        # might fail in some cases if prune=True
        assert np.allclose(
            np.unique(subdomains_cells.flatten()), np.arange(mesh.points.shape[0])
        )

        if rcm:
            if subdomains_celltype not in ("quad", "quad8", "line3"):
                raise NotImplementedError(
                    "Reversed Cuthill McKee algorithm is not implemented for "
                    + f"elements of type {subdomains_celltype}."
                )
            V = adjacency_graph(subdomains_cells, cell_type=subdomains_celltype)
            perm = reverse_cuthill_mckee(csr_matrix(V))
            self.points = points_pruned[perm, :]

            vertex_map = np.argsort(perm)
            self.cells = np.zeros_like(subdomains_cells)
            for (i, j) in np.ndindex(subdomains_cells.shape):
                self.cells[i, j] = vertex_map[subdomains_cells[i, j]]
        else:
            self.points = points_pruned
            self.cells = subdomains_cells
        self.tdim = tdim
        self.gdim = gdim
        self.cell_type = subdomains_celltype
        self._cell = cell()

    def distribute_dofs(self, dofs_per_vert, dofs_per_edge, dofs_per_face=0):
        """set number of DoFs per entity and distribute dofs

        Parameters
        ----------
        dofs_per_vert : int
            Number of DoFs per vertex.
        dofs_per_edge : int
            Number of DoFs per edge
        dofs_per_face : int, optional
            Number of DoFs per face.
        """
        self._cell.set_entity_dofs(dofs_per_vert, dofs_per_edge, dofs_per_face)
        entity_dofs = self._cell.entity_dofs
        dimension = list(range(self.tdim + 1))
        x_dofs = []
        self._dm = {dim: {} for dim in dimension}
        DoF = 0
        for ci, cell in enumerate(self.cells):
            for dim in dimension:
                # FIXME assumes cell type "quad8"
                if dim == 0:
                    entities = cell[:4]
                elif dim == 1:
                    entities = cell[4:]
                else:
                    entities = [ci]
                for local_ent, ent in enumerate(entities):
                    if ent not in self._dm[dim].keys():
                        self._dm[dim][ent] = []
                        dofs = entity_dofs[dim][local_ent]
                        for dof in dofs:
                            x_dofs.append(self.points[ent])  # store dof coordinates
                            self._dm[dim][ent].append(DoF)
                            DoF += 1
        self.n_dofs = DoF
        self.dofs_per_vert = dofs_per_vert
        self.dofs_per_edge = dofs_per_edge
        self.dofs_per_face = dofs_per_face
        self._x_dofs = np.array(x_dofs)

    def tabulate_dof_coordinates(self):
        """return dof coordinates"""
        if not hasattr(self, "_x_dofs"):
            raise AttributeError("You need to distribute DoFs first")
        return self._x_dofs

    def dofs(self):
        """return total number of dofs"""
        if not hasattr(self, "n_dofs"):
            raise AttributeError("You need to distribute DoFs first")
        return self.n_dofs

    def cell_dofs(self, cell_index):
        if not hasattr(self, "_dm"):
            raise AttributeError("You need to distribute DoFs first")
        dimension = list(range(self.tdim + 1))
        cell = self.cells[cell_index]
        cell_dofs = []
        for dim in dimension:
            if dim == 0:
                entities = cell[:4]
            elif dim == 1:
                entities = cell[4:]
            else:
                entities = [cell_index]
            for ent in entities:
                cell_dofs += self._dm[dim][ent]
        return cell_dofs

    def locate_dofs(self, X, sub=None, tol=1e-9):
        """returns dofs at coordinates X

        Parameters
        ----------
        X : list, np.ndarray
            A list of points, where each point is given as list of len(gdim).
        sub : int, optional
            Index of component.
        tol : float, optional
            Tolerance used to find coordinate.

        Returns
        -------
        dofs : list
            DoFs at given coordinates.
        """
        if isinstance(X, list):
            X = np.array(X).reshape(len(X), self.gdim)
        assert isinstance(X, np.ndarray)

        dofs = []
        for x in X:
            p = np.abs(self._x_dofs - x)
            v = np.where(np.all(p < tol, axis=1))[0]
            if v.size < 1:
                raise IndexError(f"The point {x} is not a vertex of the grid!")
            dofs += v.tolist()

        if sub is not None:
            # FIXME user has to know that things might go wrong if
            # X contains vertex AND edge coordinates ...
            dim = v.size
            return dofs[sub::dim]
        else:
            return dofs
