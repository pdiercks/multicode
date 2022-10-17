import pathlib
from dolfinx.io import gmshio
import numpy as np
from mpi4py import MPI
from multi.domain import StructuredQuadGrid
from multi.preprocessing import create_rectangle_grid, create_rce_grid_01
import tempfile


def test():
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(0., 1., 0., 1., num_cells=(10, 10), recombine=True, out_file=tf.name)
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    grid = StructuredQuadGrid(domain)
    patch_sizes = []
    for i in range(100):
        cell_patch = grid.get_patch(i)
        patch_sizes.append(cell_patch.size)
    assert np.amin(patch_sizes) == 4
    assert np.amax(patch_sizes) == 9
    num_inner = 100 - 4 - 4 * 8
    expected = num_inner * 9 + 4 * 4 + 4 * 8 * 6
    assert np.sum(patch_sizes) == expected

    num_entities = (4, 4)
    for dim, num in enumerate(num_entities):
        ents = grid.get_entities(dim, 8)
        assert ents.size == num

    verts = grid.get_entities(0, 67)  # assume cell 67 is surrounded by 8 cells
    cells_67 = grid.get_cells(0, verts)
    assert cells_67.size == 9
    assert np.allclose(cells_67, grid.get_patch(67))


def test_fine_grid_creation():
    # ### create coarse grid
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(0., 2., 0., 2., num_cells=(2, 2), recombine=True, out_file=tf.name)
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    grid = StructuredQuadGrid(domain)

    # ### create fine grid with certain rce structure
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rce_grid_01(0., 1., 0., 1., num_cells_per_edge=4, out_file=tf.name)
        rce_domain, cmarkers, fmarkers = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
        num_cells = 1
        fine_grids = np.repeat(np.array([tf.name]), num_cells)
        grid.fine_grids = fine_grids

        # target = tempfile.NamedTemporaryFile(suffix=".msh4", delete=False)
        target = pathlib.Path("/home/pdiercks/Desktop/merged.msh")
        grid.create_fine_grid(np.arange(num_cells), target.as_posix())
        print(target.name)

        # ### try to load created .msh file
        # breakpoint()
        # fine_domain, cell_markers, facet_markers = gmshio.read_from_msh(
        #         target.as_posix(), MPI.COMM_WORLD, gdim=2
        #         )


if __name__ == "__main__":
    test()
    test_fine_grid_creation()
