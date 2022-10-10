import tempfile
import meshio
import numpy as np
from multi.preprocessing import create_rce_grid_01


def test():
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=True) as tf:
        create_rce_grid_01(0., 1., 0., 1., num_cells_per_edge=10, out_file=tf.name)
        mesh = meshio.read(tf.name)
        assert "gmsh:physical" in mesh.cell_data.keys()
        assert all([np.sum(data) > 0 for data in mesh.cell_data["gmsh:physical"]])


if __name__ == "__main__":
    test()
