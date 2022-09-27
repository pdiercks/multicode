import meshio
import pathlib
import numpy as np
from multi.basis_loader import BasesLoader
from multi.domain import StructuredGrid


if __name__ == "__main__":
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
