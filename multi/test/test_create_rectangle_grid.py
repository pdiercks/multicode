from multi.preprocessing import create_rectangle_grid


if __name__ == "__main__":
    mesh = create_rectangle_grid(0.0, 1.0, 0.0, 1.0, num_cells=(2, 2))
    assert mesh.get_cells_type("triangle").shape[0] == 8

    mesh = create_rectangle_grid(0.0, 1.0, 0.0, 1.0, num_cells=(4, 4), recombine=True)
    assert mesh.get_cells_type("quad").shape[0] == 16
