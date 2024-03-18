import pytest
from mpi4py import MPI
import tempfile
import meshio
from dolfinx.io import gmshio
import numpy as np
from multi.preprocessing import create_voided_rectangle


def test_no_meshtags():
    num_cells = 10
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_voided_rectangle(0., 1., 0., 1., num_cells=num_cells, recombine=True, out_file=tf.name)
        mesh = meshio.read(tf.name)
        assert "gmsh:physical" in mesh.cell_data.keys()
        assert all([np.sum(data) > 0 for data in mesh.cell_data["gmsh:physical"]])

        domain, ct, _ = gmshio.read_from_msh(
                tf.name, MPI.COMM_WORLD, gdim=2
                )
        assert ct.topology is domain.topology
        assert ct.find(0).size == 0
        assert ct.find(3).size == 0
        assert ct.find(2).size == 0
        assert ct.find(1).size > 0

    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        with pytest.raises(ValueError):
            create_voided_rectangle(0., 1., 0., 1., num_cells=11, out_file=tf.name)


def test_physical_groups():
    num_cells = 10
    cell_tags = {"matrix": 6}
    facet_tags = {"bottom": 11, "left": 22, "right": 33, "top": 44}
    offset = {2: 0}

    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_voided_rectangle(0., 1., 0., 1., num_cells=num_cells, recombine=True, cell_tags=cell_tags, facet_tags=facet_tags, out_file=tf.name, tag_counter=offset)

        assert np.isclose(offset[2], 8)

        mesh = meshio.read(tf.name)
        assert "gmsh:physical" in mesh.cell_data.keys()
        assert all([np.sum(data) > 0 for data in mesh.cell_data["gmsh:physical"]])

        domain, ct, ft = gmshio.read_from_msh(
                tf.name, MPI.COMM_WORLD, gdim=2
                )
        assert domain.topology.dim == 2
        assert ct.find(0).size == 0
        assert ct.find(3).size == 0
        assert ct.find(2).size == 0
        assert ct.find(cell_tags["matrix"]).size > 0
        assert ft.find(facet_tags["bottom"]).size > 0
        assert ft.find(facet_tags["top"]).size > 0
        assert ft.find(facet_tags["right"]).size > 0
        assert ft.find(facet_tags["left"]).size > 0
