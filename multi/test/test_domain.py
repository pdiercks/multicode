"""test domain module"""

import tempfile
import numpy as np
from dolfinx import mesh
from mpi4py import MPI
from multi.domain import Domain, RectangularDomain
from multi.preprocessing import create_line_grid

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

    my_edges = {}
    edges = ["bottom", "right", "top", "left"]
    points = [
            ([0, 0, 0], [1, 0, 0]),
            ([1, 0, 0], [1, 1, 0]),
            ([0, 1, 0], [1, 1, 0]),
            ([0, 0, 0], [0, 1, 0]),
            ]
    for name, (start, end) in zip(edges, points):
        with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
            create_line_grid(start, end, num_cells=10, out_file=tf.name)
            line, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
            my_edges[name] = line
    
    rectangle = RectangularDomain(
        get_unit_square_mesh(10, 10), index=17, edges=my_edges
    )
    assert len(rectangle.edges.keys()) == 4
    assert isinstance(rectangle.edges["bottom"], mesh.Mesh)
    rectangle.translate([2.4, 2.4, 0.0])
    x_top = rectangle.edges["top"].geometry.x
    assert np.allclose(np.amin(x_top, axis=0), np.array([2.4, 3.4, 0.]))


if __name__ == "__main__":
    test_1d()
    test_2d()
