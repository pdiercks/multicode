import meshio
import pathlib
import numpy as np
from multi.basis_loader import BasesLoader
from multi.domain import StructuredGrid


"""test case: block.msh
cell indices of the mesh

    258
    147
    036

• 3x3 quad mesh
• load basis for each of the nine cells --> need 9 npz files
• print out bases config?
• assert number of modes
• assert shape of each basis
"""
DATA = pathlib.Path(__file__).parent / "data"


def prepare_test_data(n, modes, cell_index):
    phi = np.ones((8, n))
    bottom = np.ones((modes[0], n)) * 2
    right = np.ones((modes[1], n)) * 3
    top = np.ones((modes[2], n)) * 4
    left = np.ones((modes[3], n)) * 5
    np.savez(DATA / f"basis_{cell_index:03}.npz", phi=phi, b=bottom, r=right, t=top, l=left)


def test_bases_loader_read_bases():
    """
    258
    147
    036
    """
    V_dim = 10
    prepare_test_data(V_dim, [9, 99, 99, 9], 0)
    prepare_test_data(V_dim, [1, 99, 1, 1], 1)
    prepare_test_data(V_dim, [99, 99, 2, 2], 2)
    prerare_test_data(V_dim, [3, 3, 99, 3], 3)
    prepare_test_data(V_dim, [4, 4, 4, 4], 4)
    prepare_test_data(V_dim, [99, 5, 5, 5], 5)
    prepare_test_data(V_dim, [6, 6, 99, 99], 6)
    prepare_test_data(V_dim, [7, 7, 7, 99], 7)
    prepare_test_data(V_dim, [99,8,8,9999], 8)

    expected_num_modes = np.array([
        [9, 3, 1, 9],
        [1, 4, 1, 1],
        [1, 5, 2, 2],
        [3, 3, 4, 3],
        [4, 4, 4, 4],
        [4, 5, 5, 5],
        [6, 6, 7, 3],
        [7, 7, 7, 4],
        [7, 8, 8, 5]])

    msh_file = pathlib.Path(__file__).parent / "data/block.msh"
    mesh = meshio.read(msh_file.as_posix())

    points = mesh.points
    cells = mesh.get_cells_type("quad")
    boundary = mesh.get_cells_type("line")

    tdim = 2
    grid = StructuredGrid(points, cells, tdim)
    boundary_cells = grid.get_cells_by_points(boundary)
    inner_cells = np.setdiff1d(np.arange(len(cells)), boundary_cells)

    x_corners = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 3.0, 0.0], [0.0, 3.0, 0.0]]
    )
    corner_points = grid.get_point_tag(x_corners)  # FIXME NotImplementedError
    corner_cells = grid.get_cells_by_points(corner_points)

    grid.inner_cells = inner_cells
    grid.boundary_cells = boundary_cells
    grid.corner_cells = corner_cells

    directory = pathlib.Path("./data/")
    loader = BasesLoader(directory, grid)
    bases, modes = loader.read_bases()

    assert len(bases) == 9
    assert np.allclose(expected_num_modes, modes)
    expected = np.sum(expected_num_modes, axis=0)
    result = np.array([], dtype=int)
    for i in range(9):
        result = np.append(bases[i].shape[0])
    assert np.allclose(expected, result)



if __name__ == "__main__":
    test_bases_loader_read_bases()
