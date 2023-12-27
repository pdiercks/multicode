import tempfile
import meshio
from multi.preprocessing import create_rectangle


def test():
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=True) as tf:
        create_rectangle(0.0, 1.0, 0.0, 1.0, num_cells=(2, 2), out_file=tf.name)
        mesh = meshio.read(tf.name)
        assert mesh.get_cells_type("triangle").shape[0] == 8

    gmsh_options = {'Mesh.ElementOrder': 2, 'Mesh.SecondOrderIncomplete': 1}
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=True) as tf:
        create_rectangle(0.0, 2.0, 0.0, 2.0, num_cells=(2, 2), recombine=True, out_file=tf.name, options=gmsh_options)
        mesh = meshio.read(tf.name)
        assert mesh.get_cells_type("quad8").shape[0] == 4


if __name__ == "__main__":
    test()
