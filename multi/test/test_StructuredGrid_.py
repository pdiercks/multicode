import dolfinx
import numpy as np
from mpi4py import MPI
from multi.domain import StructuredQuadGrid


def test():
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, dolfinx.mesh.CellType.quadrilateral)
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
        ents = grid.get_cell_entities(8, dim)
        assert ents.size == num

    h = 1./10
    cell_lower_left = grid.get_cells_points(np.array([[0., 0., 0.]], dtype=np.float64))
    assert cell_lower_left.size == 1
    other = grid.get_cells_points(np.array(np.array([[h, h, 0.]], dtype=np.float64)))
    assert other.size == 4

    tags = grid.get_cell_entities(67, 0)  # assume cell 67 is surrounded by 8 cells
    cells_67 = grid.get_cells_point_tags(tags)
    assert cells_67.size == 9
    assert np.allclose(cells_67, grid.get_patch(67))


if __name__ == "__main__":
    test()
