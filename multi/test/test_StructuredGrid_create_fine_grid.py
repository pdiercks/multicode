import meshio
import numpy as np
from multi.domain import StructuredGrid


if __name__ == "__main__":
    m = meshio.read("./r2.msh")
    points = m.points
    cells = m.get_cells_type("quad")
    grid = StructuredGrid(points, cells, 2)
    fn = "./data/rce_type_01.msh"
    finegrids = np.tile(np.array([fn]), (4,))
    grid.fine_grids = finegrids
    grid.create_fine_grid(np.arange(3), "./domain.msh")
