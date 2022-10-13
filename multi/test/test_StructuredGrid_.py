from dolfinx.io import gmshio
import numpy as np
from mpi4py import MPI
from multi.domain import StructuredQuadGrid
from multi.preprocessing import create_rectangle_grid
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

    # h = 1./10
    # cell_lower_left = grid.get_cells_points(np.array([[0., 0., 0.]], dtype=np.float64))
    # assert cell_lower_left.size == 1
    # other = grid.get_cells_points(np.array(np.array([[h, h, 0.]], dtype=np.float64)))
    # FIXME not all 4 cells are returned,
    # maybe there is some kind of tolerance on the bounding boxes?
    # assert other.size == 4

    verts = grid.get_entities(0, 67)  # assume cell 67 is surrounded by 8 cells
    cells_67 = grid.get_cells(0, verts)
    assert cells_67.size == 9
    assert np.allclose(cells_67, grid.get_patch(67))


if __name__ == "__main__":
    test()
