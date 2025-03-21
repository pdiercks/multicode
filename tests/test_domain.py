"""test domain module"""

import numpy as np
import tempfile
import pytest
from dolfinx.io import gmshio
from mpi4py import MPI
from multi.boundary import within_range, plane_at
from multi.domain import Domain
from multi.preprocessing import create_line, create_rectangle, create_meshtags


def get_unit_square_mesh(nx=8, ny=8):
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(0., 1., 0., 1., num_cells=(nx, ny), recombine=True, out_file=tf.name)
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    return domain


def get_unit_interval_mesh(n):
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_line([0., 0., 0.], [1., 0., 0.], num_cells=n, out_file=tf.name)
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    return domain


def test_1d():
    domain = Domain(get_unit_interval_mesh(10))
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

    assert np.isclose(domain.num_cells, 10)


def test_2d():
    nx = ny = 8
    unit_square = get_unit_square_mesh(nx=nx, ny=ny)

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
    assert np.isclose(domain.num_cells, nx*ny)



if __name__ == "__main__":
    test_1d()
    test_2d()
