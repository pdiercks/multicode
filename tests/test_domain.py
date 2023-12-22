"""test domain module"""

import numpy as np
import tempfile
import pytest
from dolfinx import mesh
from dolfinx.io import gmshio
from mpi4py import MPI
from multi.boundary import within_range, plane_at
from multi.domain import Domain, RectangularSubdomain
from multi.preprocessing import create_line, create_rectangle, create_meshtags


def get_unit_square_mesh(nx=8, ny=8):
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(0., 1., 0., 1., num_cells=(nx, ny), recombine=True, out_file=tf.name)
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    return domain


def get_unit_interval_mesh():
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_line([0., 0., 0.], [1., 0., 0.], num_cells=10, out_file=tf.name)
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    return domain


def test_1d():
    domain = Domain(get_unit_interval_mesh())
    xmin = domain.xmin
    xmax = domain.xmax

    assert xmax[0] == 1.0
    assert np.sum(xmax) == 1.0
    assert xmin[0] == 0.0
    assert np.sum(xmin) == 0.0

    domain.translate([2.3, 4.7, 0.6])
    xmin = domain.xmin
    xmax = domain.xmax

    assert xmax[0] == 3.3
    assert np.sum(xmax) == 3.3 + 4.7 + 0.6
    assert xmin[0] == 2.3
    assert np.sum(xmin) == 2.3 + 4.7 + 0.6


def test_2d():
    unit_square = get_unit_square_mesh()

    # ### invalid MeshTags
    ct, _ = create_meshtags(unit_square, 2, {"my_subdomain": (0, within_range([0.0, 0.0], [0.5, 0.5]))})
    ft, _ = create_meshtags(unit_square, 1, {"bottom": (0, plane_at(0., "y"))})
    with pytest.raises(ValueError):
        Domain(unit_square, cell_tags=ct, facet_tags=None)
    with pytest.raises(ValueError):
        Domain(unit_square, cell_tags=None, facet_tags=ft)

    # ### translation
    domain = Domain(unit_square, cell_tags=None, facet_tags=None)
    domain.translate([2.1, 0.4, 0.0])
    xmax = domain.xmax
    assert np.isclose(xmax[1], 1.4)
    assert np.isclose(xmax[0], 3.1)

    # ### RectangularSubdomain
    rectangle = RectangularSubdomain(17, get_unit_square_mesh(10, 10))
    rectangle.create_edge_grids({"fine": 10, "coarse": 1})
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

    rectangle.create_coarse_grid(2)
    cgrid = rectangle.coarse_grid
    assert isinstance(cgrid, mesh.Mesh)
    assert cgrid.topology.dim == 2
    num_cells = cgrid.topology.index_map(cgrid.topology.dim).size_local
    assert num_cells == 4

    with pytest.raises(AttributeError):
        rectangle.create_coarse_grid(1)



if __name__ == "__main__":
    test_1d()
    test_2d()
