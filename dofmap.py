"""DofMap to handle coarse and fine scale DoFs"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import meshio
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee





def adjacency_graph(cells, cell_type="quad"):
    N = np.unique(cells).size  # number of points
    r = np.zeros((N, N), dtype=int)

    if cell_type == "quad":
        local_adj = {
            0: (1, 3),
            1: (2, 0),
            2: (3, 1),
            3: (0, 2),
        }
    else:
        raise NotImplementedError

    for cell in cells:
        for vertex, neighbours in local_adj.items():
            for n in neighbours:
                r[cell[vertex], cell[n]] = 1
    return r


# FIXME is this still up to date?
# this code uses parts of the method msh_to_xdmf() from the dolfiny project
# see here https://github.com/michalhabera/dolfiny/blob/master/dolfiny/mesh.py
class DofMap:
    """class representing a DofMap of a lagrange function space where each vertex
    is associated with gdim (or 1 if scalar) DoFs. In addition a up to 'modes'
    DoFs are defined for each cell (center), representing the bubble modes.

    Parameters
    ----------
    mshfile
        Name of .msh file (incl. extension).
    tdim : int
        Topological dimension of the mesh.
    gdim : int, optional
        Geometrical dimension of the mesh.
    scalar : bool, optional
        Whether the function space is scalar or not.
    prune_z_0 : bool, optional
        If True, prune zero z component.
    rcm : bool, optional
        If True, reorder the vertices by use of the Reverse Cuthill McKee algorithm.

    """

    def __init__(
        self, mshfile, tdim, gdim=2, scalar=False, modes=0, prune_z_0=False, rcm=False
    ):
        logger = logging.getLogger("DofMap")

        logger.info("Reading Gmsh mesh into meshio from path {}".format(mshfile))
        mesh = meshio.read(mshfile)

        if not meshio.__version__ == "4.3.1":
            raise NotImplementedError

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

        self.gdim = gdim
        self.scalar = scalar
        self.cell_type = subdomains_celltype
        self.modes = modes

    @property
    def modes(self):
        return self._modes

    @modes.setter
    def modes(self, modes_per_edge):
        if modes_per_edge < 0:
            raise ValueError
        self._modes = int(modes_per_edge)

    def dofs(self):
        """return total number of dofs"""
        return self.vertex_dofs() + self.bubble_dofs()

    def vertex_dofs(self):
        """return number of all vertex dofs"""
        dim = 1 if self.scalar else self.gdim
        number_vertices = self.points.shape[0]
        return number_vertices * dim

    def bubble_dofs(self):
        """return max number of all bubble dofs"""
        dim = self.modes
        number_cells = self.cells.shape[0]
        return number_cells * dim

    def cell_vertex_dofs(self, cell_index):
        """return vertex dofs for cell"""
        dim = 1 if self.scalar else self.gdim
        vertices = self.cells[cell_index]
        dofs = []
        for v in vertices:
            j = v * dim
            for component in range(dim):
                dofs.append(j + component)
        return np.array(dofs)

    def cell_bubble_dofs(self, cell_index):
        """return bubble dofs for cell"""
        vertex_dofs = self.vertex_dofs()
        dofs = np.arange(self.modes) + cell_index * self.modes + vertex_dofs
        return dofs

    def locate_vertex_dofs(self, X, sub=None, tol=1e-9):
        """returns dofs at vertex coordinates X

        Parameters
        ----------
        X : list, np.ndarray
            A list of points, where each point is given as list of len(gdim).
        sub : int, optional
            Index of component.
        tol : float, optional
            Tolerance used to find vertex coordinate.

        Returns
        -------
        dofs : list
            DoFs at given vertices.
        """
        if isinstance(X, list):
            X = np.array(X).reshape(len(X), self.gdim)
        assert isinstance(X, np.ndarray)
        dim = 1 if self.scalar else self.gdim

        dofs = []
        for x in X:
            p = np.abs(self.points - x)
            v = np.where(np.all(p < tol, axis=1))[0]
            if v.size < 1:
                raise IndexError(f"The point {x} is not a vertex of the grid!")
            for d in range(dim):
                dofs.append(dim * v[0] + d)

        if sub is not None:
            return dofs[sub::dim]
        else:
            return dofs
