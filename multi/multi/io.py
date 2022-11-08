import pathlib
import numpy as np
from multi.dofmap import QuadrilateralDofLayout


def read_bases(bases, modes_per_edge=None, return_num_modes=False):
    """read basis functions for multiple reduced bases

    Parameters
    ----------
    bases : list of tuple
        Each element of ``bases`` is a tuple where the first element is a
        FilePath and the second element is a string specifying which basis
        functions to load. Possible string values are 'phi' (coarse
        scale functions), and 'bottom', 'right', 'top', 'left'
        (for respective fine scale functions).
    modes_per_edge : int, optional
        Maximum number of modes per edge for the fine scale bases.
    return_num_modes : bool, optional
        If True, return number of modes per edge.

    Returns
    -------
    B : np.ndarray
        The full reduced basis.
    num_modes : tuple of int
        The maximum number of modes per edge (if ``return_num_modes`` is True).

    """
    loaded = set()
    basis_functions = {}
    num_modes = {}

    # return values
    R = []
    num_max_modes = []

    for filepath, string in bases:
        loaded.add(string)
        npz = np.load(filepath)
        try:
            basis_functions[string] = npz[string]
        except KeyError:
            basis_functions[string] = list()
        npz.close()
        num_modes[string] = len(basis_functions[string])

    dof_layout = QuadrilateralDofLayout()
    edges = [
        dof_layout.local_edge_index_map[i] for i in range(dof_layout.num_entities[1])
    ]

    max_modes_per_edge = modes_per_edge or max([num_modes[edge] for edge in edges])

    R.append(basis_functions["phi"])
    for edge in edges:
        rb = basis_functions[edge][:max_modes_per_edge]
        num_max_modes.append(len(rb))
        if len(rb) > 0:
            R.append(rb)

    if return_num_modes:
        return np.vstack(R), tuple(num_max_modes)
    else:
        return np.vstack(R)


class BasesLoader(object):
    def __init__(self, directory, coarse_grid):
        folder = pathlib.Path(directory)
        assert folder.is_dir()
        self.dir = folder
        self.grid = coarse_grid
        self.num_cells = coarse_grid.num_cells

    def read_bases(self):
        """read basis and max number of modes for each cell in the coarse grid"""
        self._build_bases_config()

        bases = []
        num_max_modes = []
        for cell_index in range(self.num_cells):
            basis, modes = read_bases(
                self._bases_config[cell_index], return_num_modes=True
            )
            bases.append(basis)
            num_max_modes.append(modes)

        max_modes = np.vstack(num_max_modes)

        return bases, max_modes

    @property
    def cell_sets(self):
        """the cell sets according to which the bases are loaded"""
        return self._cell_sets

    @cell_sets.setter
    def cell_sets(self, cell_sets):
        self._cell_sets = {}
        assert "inner" in cell_sets.keys()
        # sort cell indices in increasing order
        for key, value in cell_sets.items():
            self._cell_sets[key] = np.sort(list(value))

    def _build_bases_config(self):
        """builds logic to read (edge) bases such that a conforming global approx results"""

        marked_edges = {}
        self._bases_config = {}
        dof_layout = QuadrilateralDofLayout()

        try:
            cell_sets = self.cell_sets
        except AttributeError as err:
            raise err("You have to define `cell_sets` to load bases functions.")

        for cset in cell_sets.values():
            for cell_index in cset:

                path = self.dir / f"basis_{cell_index:03}.npz"
                self._bases_config[cell_index] = []
                self._bases_config[cell_index].append((path, "phi"))

                edges = self.grid.get_entities(1, cell_index)
                for local_ent, ent in enumerate(edges):
                    edge = dof_layout.local_edge_index_map[local_ent]
                    if ent not in marked_edges.keys():
                        # edge is 'visited' the first time
                        # add edge to be loaded from current path
                        self._bases_config[cell_index].append((path, edge))

                        # mark edge as 'visited'
                        marked_edges[ent] = path
                    else:
                        # edge was already visited
                        # retrieve `path` from previously marked edges
                        self._bases_config[cell_index].append((marked_edges[ent], edge))
        assert len(self._bases_config.keys()) == self.num_cells
