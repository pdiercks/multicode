import shutil
import tempfile
from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio
from multi.preprocessing import create_rectangle
from multi.domain import StructuredQuadGrid
from multi.io import select_modes, BasesLoader



def test_mode_selection():
    num_coarse_modes = 8
    num_modes_per_edge = 2
    num_modes = num_coarse_modes + num_modes_per_edge * 4
    dim = 11
    basis = np.repeat(np.ones((1, dim)), repeats=num_modes, axis=0)
    bottom_0 = (8, np.ones(dim) * 12.3)
    basis[bottom_0[0]] = bottom_0[1]

    num_active = 1
    my_modes = select_modes(basis, num_modes_per_edge, num_active)
    assert np.isclose(len(my_modes), num_coarse_modes + 4 * num_active)
    assert np.isclose(np.sum(my_modes), dim * num_coarse_modes + dim * 3 * num_active + dim * 12.3)

    other = select_modes(basis, num_modes_per_edge, [3, 0, 0, 1])
    assert np.isclose(np.sum(other), dim * num_coarse_modes + dim * (12.3 + 1. + 1.))


def prepare_test_data(n, modes, filepath):
    phi = np.ones((8, n))
    bottom = np.ones((modes[0], n)) * 2
    right = np.ones((modes[2], n)) * 3
    top = np.ones((modes[3], n)) * 4
    left = np.ones((modes[1], n)) * 5
    np.savez(
        filepath.as_posix(), phi=phi, bottom=bottom, right=right, top=top, left=left
    )


def test_bases_loader_read_bases():
    """
    23
    01
    """
    TEST = Path(__file__).parent
    DATA = TEST / "data"
    if not DATA.exists():
        DATA.mkdir()

    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(
            0.0,
            3.0,
            0.0,
            3.0,
            num_cells=(2, 2),
            facet_tags={"bottom": 1, "left": 2, "right": 3, "top": 4},
            recombine=True,
            out_file=tf.name,
        )
        domain, cell_markers, facet_markers = gmshio.read_from_msh(
            tf.name, MPI.COMM_WORLD, gdim=2
        )

    grid = StructuredQuadGrid(domain, cell_markers, facet_markers)
    num_cells = grid.num_cells
    assert num_cells == 4

    V_dim = 10
    # prepare data for each coarse grid cell
    # space_dim, modes (bottom, left, right, top), cell_index
    for cell_index in range(num_cells):
        n = cell_index + 1
        num_modes = [n, ] * 4
        prepare_test_data(V_dim, num_modes, DATA / f"basis_{cell_index:03}.npz")

    loader = BasesLoader(DATA, num_cells)
    bases, modes = loader.read_bases()

    # check on number of modes
    assert np.isclose(np.sum(modes) + num_cells * 8, np.sum([len(rb) for rb in bases]))

    # check values for some
    assert np.isclose(np.sum(bases[0]), V_dim * 8 + V_dim * (2 + 3 + 4 + 5))
    assert np.isclose(np.sum(bases[1]), V_dim * 8 + 2 * V_dim * (2 + 3 + 4 + 5))
    shutil.rmtree(DATA)
