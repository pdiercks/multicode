"""test domain module"""

import numpy as np
from dolfinx import mesh
from mpi4py import MPI
from multi.domain import Domain, RceDomain

from dolfinx.io import gmshio


def get_unit_square_mesh(nx=8, ny=8):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
    return domain


def get_unit_interval_mesh():
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 10)
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

    xmin = domain.xmin
    xmax = domain.xmax

    assert np.isclose(xmax[1], 1.4)
    assert np.isclose(xmax[0], 3.1)

    rectangle = RceDomain(
        get_unit_square_mesh(10, 10), index=17, edges=True
    )
    assert len(rectangle.edges.keys()) == 4
    assert isinstance(rectangle.edges["bottom"][0], mesh.Mesh)
    assert isinstance(rectangle.edges["top"][0], mesh.Mesh)
    assert isinstance(rectangle.edges["right"][0], mesh.Mesh)
    assert isinstance(rectangle.edges["left"][0], mesh.Mesh)
    rectangle.translate([2.4, 2.4, 0.0])
    x_top = rectangle.edges["top"][0].geometry.x
    assert np.allclose(np.amin(x_top, axis=0), np.array([2.4, 3.4, 0.]))


if __name__ == "__main__":
    test_1d()
    test_2d()
