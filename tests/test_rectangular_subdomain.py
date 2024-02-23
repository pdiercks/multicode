import tempfile
import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import mesh
from dolfinx.io import gmshio
from multi.preprocessing import create_rectangle, create_voided_rectangle
from multi.domain import RectangularSubdomain


@pytest.mark.parametrize("create_grid", [create_rectangle, create_voided_rectangle])
def test(create_grid):
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
            create_grid(
                0.0,
                1.0,
                0.0,
                1.0,
                num_cells=10,
                recombine=True,
                out_file=tf.name,
            )
            domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    rectangle = RectangularSubdomain(17, domain)
    rectangle.create_coarse_grid(1)
    rectangle.create_boundary_grids()
    assert len(rectangle.fine_edge_grid.keys()) == 4
    assert len(rectangle.coarse_edge_grid.keys()) == 4
    assert isinstance(rectangle.fine_edge_grid["bottom"], mesh.Mesh)
    assert isinstance(rectangle.fine_edge_grid["top"], mesh.Mesh)
    assert isinstance(rectangle.fine_edge_grid["right"], mesh.Mesh)
    assert isinstance(rectangle.fine_edge_grid["left"], mesh.Mesh)
    assert isinstance(rectangle.coarse_edge_grid["bottom"], mesh.Mesh)
    assert isinstance(rectangle.coarse_edge_grid["top"], mesh.Mesh)
    assert isinstance(rectangle.coarse_edge_grid["right"], mesh.Mesh)
    assert isinstance(rectangle.coarse_edge_grid["left"], mesh.Mesh)

    rectangle = RectangularSubdomain(18, domain)
    rectangle.create_coarse_grid(2)
    cgrid = rectangle.coarse_grid
    assert isinstance(cgrid, mesh.Mesh)
    assert cgrid.topology.dim == 2
    num_cells = cgrid.topology.index_map(cgrid.topology.dim).size_local
    assert num_cells == 4

    bottom = rectangle.str_to_marker("bottom")
    top = rectangle.str_to_marker("top")
    left = rectangle.str_to_marker("left")
    right = rectangle.str_to_marker("right")
    origin = np.array([[0.0], [0.0], [0.0]])
    assert bottom(origin)
    assert left(origin)
    assert not top(origin)
    assert not right(origin)

    with pytest.raises(AttributeError):
        rectangle.create_coarse_grid(1)

    with pytest.raises(ValueError):
        rectangle.str_to_marker("valueerror")
