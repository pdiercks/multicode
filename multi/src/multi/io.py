import pathlib
import numpy as np
from multi.dofmap import QuadrilateralDofLayout


def select_modes(basis, max_modes, active_modes):
    """select modes according to local dof layout and currently active
    number of modes

    Parameters
    ----------
    basis : np.ndarray
        The multiscale basis used.
    max_modes : int or list of int
        Maximum number of modes per edge.
    active_modes : int or list of int
        Number of modes per edge to be used.

    Returns
    -------
    basis : np.ndarray
        Subset of the full basis.

    """
    if isinstance(max_modes, (int, np.integer)):
        max_modes = [max_modes] * 4
    if isinstance(active_modes, (int, np.integer)):
        active_modes = [active_modes] * 4

    # make sure that active_modes[edge] <= max_modes[edge]
    assert len(max_modes) == len(active_modes)
    for i in range(len(max_modes)):
        if active_modes[i] > max_modes[i]:
            active_modes[i] = max_modes[i]

    dof_layout = QuadrilateralDofLayout()
    edges = [
        dof_layout.local_edge_index_map[i] for i in range(dof_layout.num_entities[1])
    ]

    gdim = 2  # FIXME better get this from dofmap ...
    coarse = np.arange(gdim * dof_layout.num_entities[0], dtype=np.int32)
    offset = coarse.size
    index_map = {"phi": coarse}

    mask = []
    mask.append(coarse)

    for edge in edges:
        index = dof_layout.local_edge_index_map[edge]

        index_map[edge] = np.arange(max_modes[index], dtype=np.int32) + offset
        offset += index_map[edge].size
        selected = np.arange(active_modes[index], dtype=np.int32)

        mask.append(index_map[edge][selected])

    mask = np.hstack(mask)
    return basis[mask]


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
    def __init__(self, directory, num_cells):
        folder = pathlib.Path(directory)
        assert folder.is_dir()
        self.dir = folder
        self.num_cells = num_cells

    def read_bases(self):
        """read basis and max number of modes
        for each cell in the coarse grid"""
        self._build_bases_config()

        bases = []
        num_max_modes = []
        for cell_index in range(self.num_cells):
            basis, modes = read_bases(
                self._config[cell_index], return_num_modes=True
            )
            bases.append(basis)
            num_max_modes.append(modes)

        max_modes = np.vstack(num_max_modes)

        return bases, max_modes

    def _build_bases_config(self):
        cfg = {}
        for ci in range(self.num_cells):
            cfg[ci] = []
            path = self.dir / f"basis_{ci:03}.npz"
            for basis in ["phi", "bottom", "left", "right", "top"]:
                cfg[ci].append((path, basis))
        self._config = cfg

    # @property
    # def cell_sets(self):
    #     """the cell sets according to which the bases are loaded"""
    #     return self._cell_sets

    # @cell_sets.setter
    # def cell_sets(self, cell_sets):
    #     self._cell_sets = {}
    #     assert "inner" in cell_sets.keys()
    #     # sort cell indices in increasing order
    #     for key, value in cell_sets.items():
    #         self._cell_sets[key] = np.sort(list(value))

    # def _build_bases_config(self):
    #     """builds logic to read (edge) bases
    #     such that a conforming global approx results"""

    #     marked_edges = {}
    #     self._bases_config = {}
    #     dof_layout = QuadrilateralDofLayout()

    #     try:
    #         cell_sets = self.cell_sets
    #     except AttributeError as err:
    #         raise err("You have to define `cell_sets` to load bases functions.")

    #     for cset in cell_sets.values():
    #         for cell_index in cset:

    #             path = self.dir / f"basis_{cell_index:03}.npz"
    #             self._bases_config[cell_index] = []
    #             self._bases_config[cell_index].append((path, "phi"))

    #             edges = self.grid.get_entities(1, cell_index)
    #             for local_ent, ent in enumerate(edges):
    #                 edge = dof_layout.local_edge_index_map[local_ent]
    #                 if ent not in marked_edges.keys():
    #                     # edge is 'visited' the first time
    #                     # add edge to be loaded from current path
    #                     self._bases_config[cell_index].append((path, edge))

    #                     # mark edge as 'visited'
    #                     marked_edges[ent] = path
    #                 else:
    #                     # edge was already visited
    #                     # retrieve `path` from previously marked edges
    #                     self._bases_config[cell_index].append((marked_edges[ent], edge))
    #     assert len(self._bases_config.keys()) == self.num_cells
