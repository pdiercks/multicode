import tempfile
import meshio
from multi.preprocessing import create_line_grid


def test():
    num_cells = 10
    points = [
            ([0, 0, 0], [1, 0, 0]),
            ([0, 0, 0], [-1, 0, 0]),
            ([0, 0, 0], [0, 1, 0]),
            ([0.2, 0, 0], [1.2, 1, 0]),
            ([0.2, 0.5, 0], [-1.2, -0.3, 0]),
            ([1.2, 0.1, 0], [0.2, 1, 0])
            ]
    for start, end in points:
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=True) as tf:
            create_line_grid(start, end, num_cells=num_cells, out_file=tf.name)
            mesh = meshio.read(tf.name)
            assert mesh.points.shape == (num_cells+1, 3)
            assert mesh.get_cells_type("line").shape[0] == num_cells


if __name__ == "__main__":
    test()
