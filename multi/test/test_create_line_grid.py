from multi.preprocessing import create_line_grid


if __name__ == "__main__":
    num_cells = 10
    mesh = create_line_grid([0, 0, 0], [1, 0, 0], num_cells=num_cells)
    assert mesh.points.shape == (num_cells+1, 3)
    assert mesh.get_cells_type("line").shape[0] == num_cells

    mesh = create_line_grid([0, 0, 0], [-1, 0, 0], num_cells=num_cells)
    assert mesh.points.shape == (num_cells+1, 3)
    assert mesh.get_cells_type("line").shape[0] == num_cells

    mesh = create_line_grid([0, 0, 0], [0, 1, 0], num_cells=num_cells)
    assert mesh.points.shape == (num_cells+1, 3)
    assert mesh.get_cells_type("line").shape[0] == num_cells

    mesh = create_line_grid([0.2, 0, 0], [1.2, 1, 0], num_cells=num_cells)
    assert mesh.points.shape == (num_cells+1, 3)
    assert mesh.get_cells_type("line").shape[0] == num_cells

    mesh = create_line_grid([0.2, 0.5, 0], [-1.2, -0.3, 0], num_cells=num_cells)
    assert mesh.points.shape == (num_cells+1, 3)
    assert mesh.get_cells_type("line").shape[0] == num_cells

    mesh = create_line_grid([1.2, 0.1, 0], [0.2, 1, 0], num_cells=num_cells)
    assert mesh.points.shape == (num_cells+1, 3)
    assert mesh.get_cells_type("line").shape[0] == num_cells
