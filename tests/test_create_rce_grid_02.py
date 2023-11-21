import tempfile
import meshio
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from multi.preprocessing import create_rce_grid_02


def test():
    num_cells = 10
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rce_grid_02(0., 20., 0., 20., num_cells=num_cells, facets=True, out_file=tf.name)
        mesh = meshio.read(tf.name)
        assert "gmsh:physical" in mesh.cell_data.keys()
        assert all([np.sum(data) > 0 for data in mesh.cell_data["gmsh:physical"]])

        domain, cell_markers, facet_markers = gmshio.read_from_msh(
                tf.name, MPI.COMM_WORLD, gdim=2
                )
        assert cell_markers.topology is domain.topology
        assert cell_markers.find(0).size == 0
        assert cell_markers.find(3).size == 0
        assert cell_markers.find(1).size > 0
        assert cell_markers.find(2).size > 0
        assert facet_markers.find(1).size == num_cells
        assert facet_markers.find(2).size == num_cells
        assert facet_markers.find(3).size == num_cells
        assert facet_markers.find(4).size == num_cells


if __name__ == "__main__":
    test()
