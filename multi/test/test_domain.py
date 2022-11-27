"""test domain module"""

import numpy as np
import tempfile
import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
from multi.domain import Domain, RectangularDomain
from multi.preprocessing import create_line_grid, create_rectangle_grid


def get_unit_square_mesh(nx=8, ny=8):
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(0., 1., 0., 1., num_cells=(nx, ny), recombine=True, out_file=tf.name)
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    return domain


def get_unit_interval_mesh():
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_line_grid([0., 0., 0.], [1., 0., 0.], num_cells=10, out_file=tf.name)
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
    domain = Domain(get_unit_square_mesh())
    domain.translate([2.1, 0.4, 0.0])

    xmax = domain.xmax

    assert np.isclose(xmax[1], 1.4)
    assert np.isclose(xmax[0], 3.1)

    rectangle = RectangularDomain(
        get_unit_square_mesh(10, 10), index=17
    )
    rectangle.create_edge_grids(10)
    assert len(rectangle.fine_edge_grid.keys()) == 4
    assert len(rectangle.coarse_edge_grid.keys()) == 4
    assert isinstance(rectangle.fine_edge_grid["bottom"], dolfinx.mesh.Mesh)
    assert isinstance(rectangle.fine_edge_grid["top"], dolfinx.mesh.Mesh)
    assert isinstance(rectangle.fine_edge_grid["right"], dolfinx.mesh.Mesh)
    assert isinstance(rectangle.fine_edge_grid["left"], dolfinx.mesh.Mesh)
    assert isinstance(rectangle.coarse_edge_grid["bottom"], dolfinx.mesh.Mesh)
    assert isinstance(rectangle.coarse_edge_grid["top"], dolfinx.mesh.Mesh)
    assert isinstance(rectangle.coarse_edge_grid["right"], dolfinx.mesh.Mesh)
    assert isinstance(rectangle.coarse_edge_grid["left"], dolfinx.mesh.Mesh)


if __name__ == "__main__":
    test_1d()
    test_2d()
