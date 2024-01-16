import pytest
import tempfile
import meshio
from multi.preprocessing import create_line


@pytest.mark.parametrize("degree,cell_type,num_points", [(1, "line", 11), (2, "line3", 21)])
def test(degree, cell_type, num_points):
    num_cells = 10
    points = [
            ([0, 0, 0], [1, 0, 0]),
            ([0, 0, 0], [-1, 0, 0]),
            ([0, 0, 0], [0, 1, 0]),
            ([0.2, 0, 0], [1.2, 1, 0]),
            ([0.2, 0.5, 0], [-1.2, -0.3, 0]),
            ([1.2, 0.1, 0], [0.2, 1, 0])
            ]
    options = {"Mesh.ElementOrder": degree}
    for start, end in points:
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=True) as tf:
            create_line(start, end, num_cells=num_cells, out_file=tf.name, options=options)
            mesh = meshio.read(tf.name)
            assert mesh.points.shape == (num_points, 3)
            assert mesh.get_cells_type(cell_type).shape[0] == num_cells


if __name__ == "__main__":
    test(1, "line", 11)
