import pytest
import tempfile
import meshio
from multi.preprocessing import create_rectangle


@pytest.mark.parametrize("degree,cell_type,num_cells", [(1, "triangle", 8), (2, "triangle6", 8)])
def test_triangle(degree, cell_type, num_cells):
    options = {'Mesh.ElementOrder': degree}
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=True) as tf:
        create_rectangle(0.0, 1.0, 0.0, 1.0, num_cells=(2, 2), out_file=tf.name, options=options)
        mesh = meshio.read(tf.name)
        assert mesh.get_cells_type(cell_type).shape[0] == num_cells


@pytest.mark.parametrize("degree,cell_type,num_cells", [(1, "quad", 4), (2, "quad9", 4)])
def test_quadrilateral(degree, cell_type, num_cells):
    options = {'Mesh.ElementOrder': degree}
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=True) as tf:
        create_rectangle(0.0, 1.0, 0.0, 1.0, num_cells=(2, 2), recombine=True, out_file=tf.name, options=options)
        mesh = meshio.read(tf.name)
        assert mesh.get_cells_type(cell_type).shape[0] == num_cells
