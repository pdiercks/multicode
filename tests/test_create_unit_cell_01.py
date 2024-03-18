import pytest
from mpi4py import MPI
import tempfile
import meshio
from dolfinx.io import gmshio
import numpy as np
from multi.preprocessing import create_unit_cell_01


def test_no_tags():
    num_cells = 10
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_unit_cell_01(
            0.0,
            1.0,
            0.0,
            1.0,
            num_cells=num_cells,
            recombine=True,
            out_file=tf.name,
        )
        mesh = meshio.read(tf.name)
        assert "gmsh:physical" in mesh.cell_data.keys()
        assert all([np.sum(data) > 0 for data in mesh.cell_data["gmsh:physical"]])

        domain, ct, ft = gmshio.read_from_msh(
            tf.name, MPI.COMM_WORLD, gdim=2
        )
        assert domain.topology.dim == 2
        assert ct.values.size > 0
        assert ft.values.size == 0


def test_physical_groups():
    num_cells = 10
    cell_tags = {"matrix": 11, "inclusion": 23}
    facet_tags = {"bottom": 1, "left": 2, "right": 3, "top": 4}
    offset = {2: 0}
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_unit_cell_01(
            0.0,
            1.0,
            0.0,
            1.0,
            num_cells=num_cells,
            recombine=True,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            out_file=tf.name,
            tag_counter=offset,
        )
        # 9 surfaces are created
        assert np.isclose(offset[2], 9)

        mesh = meshio.read(tf.name)
        assert "gmsh:physical" in mesh.cell_data.keys()
        assert all([np.sum(data) > 0 for data in mesh.cell_data["gmsh:physical"]])

        domain, cell_markers, facet_markers = gmshio.read_from_msh(
            tf.name, MPI.COMM_WORLD, gdim=2
        )
        assert cell_markers.topology is domain.topology
        assert cell_markers.find(0).size == 0
        assert cell_markers.find(3).size == 0
        assert cell_markers.find(cell_tags["matrix"]).size > 0
        assert cell_markers.find(cell_tags["inclusion"]).size > 0
        assert facet_markers.find(facet_tags["bottom"]).size == num_cells
        assert facet_markers.find(facet_tags["left"]).size == num_cells
        assert facet_markers.find(facet_tags["right"]).size == num_cells
        assert facet_markers.find(facet_tags["top"]).size == num_cells

    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        with pytest.raises(ValueError):
            create_unit_cell_01(0.0, 1.0, 0.0, 1.0, num_cells=11, out_file=tf.name)
