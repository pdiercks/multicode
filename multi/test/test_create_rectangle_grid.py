from multi.preprocessing import create_rectangle_grid


if __name__ == "__main__":
    mesh = create_rectangle_grid(0.0, 1.0, 0.0, 1.0, num_cells=(2, 2), out_file="./r1.msh")
    assert mesh.get_cells_type("triangle").shape[0] == 8

    mesh = create_rectangle_grid(0.0, 2.0, 0.0, 2.0, num_cells=(2, 2), recombine=True, out_file="./r2.msh", options={"Mesh.ElementOrder": 2, "Mesh.SecondOrderIncomplete": 1})
    assert mesh.get_cells_type("quad8").shape[0] == 4
