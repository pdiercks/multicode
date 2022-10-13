import tempfile
from dolfinx.io import gmshio
from mpi4py import MPI
import pathlib
import numpy as np
from multi.preprocessing import create_rectangle_grid
from multi.io import BasesLoader
from multi.domain import StructuredQuadGrid

TEST = pathlib.Path(__file__).parent
DATA = TEST / "data"


def prepare_test_data(n, modes, cell_index):
    phi = np.ones((8, n))
    bottom = np.ones((modes[0], n)) * 2
    right = np.ones((modes[1], n)) * 3
    top = np.ones((modes[2], n)) * 4
    left = np.ones((modes[3], n)) * 5
    np.savez(
        DATA / f"basis_{cell_index:03}.npz", phi=phi, b=bottom, r=right, t=top, l=left
    )


def test_bases_loader_read_bases():
    """
    578
    246
    013
    """
    V_dim = 10
    prepare_test_data(V_dim, [1, 1, 1, 1], 0)
    prepare_test_data(V_dim, [2, 2, 2, 2], 1)
    prepare_test_data(V_dim, [3, 3, 3, 3], 2)
    prepare_test_data(V_dim, [4, 4, 4, 4], 3)
    prepare_test_data(V_dim, [5, 5, 5, 5], 4)
    prepare_test_data(V_dim, [6, 6, 6, 6], 5)
    prepare_test_data(V_dim, [7, 7, 7, 7], 6)
    prepare_test_data(V_dim, [8, 8, 8, 8], 7)
    prepare_test_data(V_dim, [9, 9, 9, 9], 8)

    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(
            0.0, 3.0, 0.0, 3.0, num_cells=(3, 3), recombine=True, out_file=tf.name
        )
        domain, cell_markers, facet_markers = gmshio.read_from_msh(
            tf.name, MPI.COMM_WORLD, gdim=2
        )

    grid = StructuredQuadGrid(domain, cell_markers, facet_markers)

    expected_num_modes = np.array(
        [
            [1, 1, 2, 3],
            [2, 2, 2, 5],
            [3, 3, 5, 3],
            [4, 2, 4, 7],
            [5, 5, 5, 5],
            [3, 6, 8, 6],
            [7, 5, 7, 7],
            [5, 8, 8, 8],
            [7, 8, 9, 9],
        ]
    )

    num_cells = grid.num_cells
    assert num_cells == 9
    # cell sets
    bottom = grid.get_cells(1, grid.facet_markers.find(1))
    left = grid.get_cells(1, grid.facet_markers.find(2))
    right = grid.get_cells(1, grid.facet_markers.find(3))
    top = grid.get_cells(1, grid.facet_markers.find(4))

    boundary_cells = np.unique(np.hstack((bottom, left, right, top)))
    inner_cells = np.setdiff1d(np.arange(num_cells), boundary_cells)
    corner_cells = np.array([0, 3, 5, 8], dtype=np.intc)
    boundary_cells = np.setdiff1d(boundary_cells, corner_cells)

    grid.cell_sets = {
        "inner": inner_cells,
        "boundary": boundary_cells,
        "corner": corner_cells,
    }
    # FIXME for the BasesLoader to work always require
    # cell sets 'inner', 'boundary' and 'corner' to be defined ...

    loader = BasesLoader(DATA, grid)
    bases, modes = loader.read_bases()

    assert len(bases) == 9
    assert np.allclose(expected_num_modes, modes)
    expected = np.sum(expected_num_modes, axis=1)
    result = np.array([], dtype=int)
    for i in range(9):
        result = np.append(result, bases[i].shape[0])
    assert np.allclose(expected, result - 8)


if __name__ == "__main__":
    test_bases_loader_read_bases()
