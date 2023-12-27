from typing import Union, Optional
from pathlib import Path
import numpy as np
import numpy.typing as npt
from multi.dofmap import QuadrilateralDofLayout


def select_modes(basis: npt.NDArray, max_modes: Union[int, list[int]], active_modes: Union[int, list[int]]) -> npt.NDArray:
    """Selects modes according to multi.dofmap.QuadrilateralDofLayout.

    Args:
        basis: The full multiscale basis.
        max_modes: The maximum number of modes per edge.
        active_modes: The number of modes per edge to be selected.

    Returns:
        modes: A subset of the full multiscale basis.

    Note:
        It assumes that the max. number of functions for each set (phi, bottom,
        right, top, left) of the multiscale basis are given in order
        compliant to multi.dofmap.QuadrilateralDofLayout.
        See also ``BasesLoader.read_bases()``.
        This function is mainly useful for selection of active modes for the
        local basis for a particular cell:

        for cell_index in range(coarse_grid.num_cells):
            local_basis = select_modes(bases[cell_index], max_modes[cell_index], dofs_per_edge[cell_index])

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


def read_bases(bases: list[tuple[Path, str]], modes_per_edge: Optional[int] = None, return_num_modes: Optional[bool] = False):
    """Reads basis functions for multiple reduced bases.

    Args:
        bases: Define data to be read (filepath) for each set given as string.
        Possible string values are 'phi', 'bottom', 'right', 'top', 'left'.
        modes_per_edge: Maximum number of modes per edge for the fine scale bases.
        return_num_modes: If True, return number of modes per edge.

    Returns:
        B: The full multiscale basis.
        num_modes: The maximum number of modes per edge if `return_num_modes` is True.

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
    def __init__(self, directory: Union[str, Path], num_cells: int):
        folder = Path(directory)
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
