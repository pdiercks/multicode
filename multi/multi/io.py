import numpy as np


def read_bases(bases, modes_per_edge=None, return_num_modes=False):
    """read basis functions for multiple reduced bases

    Parameters
    ----------
    bases : list of tuple
        Each element of ``bases`` is a tuple where the first element is a
        FilePath and the second element is a string specifying which basis
        functions to load. Possible string values are 'phi' (coarse
        scale functions), and 'b', 'r', 't', 'l' (for respective fine scale
        functions).
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
        basis_functions[string] = npz[string]
        npz.close()
        num_modes[string] = len(basis_functions[string])

    # check for completeness
    assert not len(loaded.difference(["phi", "b", "r", "t", "l"]))
    loaded.discard("phi")

    max_modes_per_edge = modes_per_edge or max([num_modes[edge] for edge in loaded])

    R.append(basis_functions["phi"])
    for edge in ["b", "r", "t", "l"]:
        rb = basis_functions[edge][:max_modes_per_edge]
        num_max_modes.append(rb.shape[0])
        R.append(rb)

    if return_num_modes:
        return np.vstack(R), tuple(num_max_modes)
    else:
        return np.vstack(R)


class BasesLoader(object):
    def __init__(self, directory, coarse_grid):
        assert directory.is_dir()
        assert coarse_grid._cell.cell_type in ("quad8", "quad9")
        assert hasattr(coarse_grid, "cell_sets")
        self.dir = directory
        self.grid = coarse_grid
        self.num_cells = coarse_grid.cells.shape[0]

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

    def _build_bases_config(self):
        """builds logic to read (edge) bases such that a conforming global approx results"""

        marked_edges = {}
        self._bases_config = {}
        int_to_edge_str = ["b", "r", "t", "l"]

        cells = self.grid.cells
        cell_sets = self.grid.cell_sets

        for cset in cell_sets.values():
            for cell_index, cell in zip(cset, cells[cset]):

                path = self.dir / f"basis_{cell_index:03}.npz"
                self._bases_config[cell_index] = []
                self._bases_config[cell_index].append((path, "phi"))

                self.grid._cell.set_entities(
                    cell
                )  # FIXME not needed if dolfinx.mesh is used
                edges = self.grid._cell.get_entities(
                    dim=1
                )  # TODO implement grid.get_cell_entities(dim) using mesh.topology.connectivity
                for local_ent, ent in enumerate(edges):
                    edge = int_to_edge_str[local_ent]
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
