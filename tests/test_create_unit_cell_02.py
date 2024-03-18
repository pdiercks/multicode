import tempfile
import meshio
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from multi.preprocessing import create_unit_cell_02


def test_no_meshtags():
    num_cells = 10
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_unit_cell_02(2.0, 22.0, 2.0, 22.0, num_cells=num_cells, out_file=tf.name)
        mesh = meshio.read(tf.name)
        assert "gmsh:physical" in mesh.cell_data.keys()
        assert all([np.sum(data) > 0 for data in mesh.cell_data["gmsh:physical"]])

        domain, ct, ft = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
        assert ct.topology is domain.topology
        assert ct.find(0).size == 0
        assert ct.find(3).size == 0
        assert ct.find(1).size > 0
        assert ct.find(2).size > 0
        assert ft.values.size == 0


def test_physical_groups():
    num_cells = 10
    cell_tags = {"matrix": 11, "aggregates": 23}
    facet_tags = {"bottom": 1, "left": 2, "right": 3, "top": 4}
    offset = {2: 0}
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_unit_cell_02(
            0.0,
            20.0,
            0.0,
            20.0,
            num_cells=num_cells,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            out_file=tf.name,
            tag_counter=offset
        )
        # 9 surfaces are created
        assert np.isclose(offset[2], 9)

        mesh = meshio.read(tf.name)
        assert "gmsh:physical" in mesh.cell_data.keys()
        assert all([np.sum(data) > 0 for data in mesh.cell_data["gmsh:physical"]])

        domain, ct, ft = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
        assert ct.topology is domain.topology
        assert ct.find(0).size == 0
        assert ct.find(3).size == 0
        assert ct.find(cell_tags["matrix"]).size > 0
        assert ct.find(cell_tags["aggregates"]).size > 0
        assert ft.find(facet_tags["bottom"]).size == num_cells
        assert ft.find(facet_tags["left"]).size == num_cells
        assert ft.find(facet_tags["right"]).size == num_cells
        assert ft.find(facet_tags["top"]).size == num_cells
