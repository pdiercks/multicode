from multi.preprocessing import create_rectangle_grid


if __name__ == "__main__":
    num_cells = (4, 4)
    mesh = create_rectangle_grid([0, 0, 0], [1, 1, 0], num_cells=num_cells)
    breakpoint()
