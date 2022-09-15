import meshio
from IPython import embed
from multi.preprocessing import create_rce_grid_01


if __name__ == "__main__":
    grid = create_rce_grid_01(0., 1., 0., 1., num_cells_per_edge=2)
    # TODO assert
