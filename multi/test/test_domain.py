"""test domain module"""

import numpy as np
from dolfinx import mesh
from mpi4py import MPI
from multi.domain import Domain


def get_unit_square_mesh():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
    return domain


def get_unit_interval_mesh():
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    return domain


# TODO parallel?
def test_1d():
    domain = Domain(get_unit_interval_mesh())
    xmin = domain.xmin()
    xmax = domain.xmax()

    assert xmax[0] == 1.0
    assert np.sum(xmax) == 1.0
    assert xmin[0] == 0.0
    assert np.sum(xmin) == 0.0

    domain.translate([2.3, 4.7, 0.6])
    xmin = domain.xmin()
    xmax = domain.xmax()

    assert xmax[0] == 3.3
    assert np.sum(xmax) == 3.3 + 4.7 + 0.6
    assert xmin[0] == 2.3
    assert np.sum(xmin) == 2.3 + 4.7 + 0.6


# TODO parallel?
def test_2d():
    domain = Domain(get_unit_square_mesh())
    domain.translate([2.1, 0.4, 0.0])

    xmin = domain.xmin()
    xmax = domain.xmax()

    assert np.isclose(xmax[1], 1.4)
    assert np.isclose(xmax[0], 3.1)

    # TODO
    # another = RectangularDomain(
    #     "data/rcedomain.xdmf", _id=2, subdomains=True, edges=True
    # )
    # assert len(another.edges) == 4
    # assert all([isinstance(e, df.cpp.mesh.Mesh) for e in another.edges])
    # assert np.sum(another.subdomains.array()) > 1
    # # subdomain numbering is assumed to start with 1 (pygmsh default)
    # Ω_i = np.amin(another.subdomains.array())
    # assert Ω_i > 0 and Ω_i < 2
    # assert np.isclose(another.xmin, 0.0)
    # assert np.isclose(another.ymin, 0.0)
    # assert np.isclose(another.xmax, another.ymax)


if __name__ == "__main__":
    test_1d()
    test_2d()
