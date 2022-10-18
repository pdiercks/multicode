import pathlib
from dolfinx.io import gmshio
import numpy as np
from mpi4py import MPI
from multi.domain import StructuredQuadGrid
from multi.preprocessing import create_rectangle_grid, create_rce_grid_01
import tempfile


def test():
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(
            0.0, 1.0, 0.0, 1.0, num_cells=(10, 10), recombine=True, out_file=tf.name
        )
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
        create_rectangle_grid(
            0.0, 2.0, 0.0, 2.0, num_cells=(2, 2), recombine=True, out_file=tf.name
        )
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    grid = StructuredQuadGrid(domain)
    grid.fine_grid_method = create_rce_grid_01

    target = pathlib.Path("/tmp/merged.msh")
    mesh, ct = grid.create_fine_grid(
        np.array([0, 1]), target.as_posix(), cell_type="triangle", num_cells=4
    )
    assert ct.find(1).size > 0
    assert ct.find(2).size > 0


if __name__ == "__main__":
    test()
    test_fine_grid_creation()
