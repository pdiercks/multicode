import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import mesh
from multi.boundary import line_at

NCELLS = 4


@pytest.mark.parametrize("domain",[
    mesh.create_unit_square(MPI.COMM_WORLD, NCELLS, NCELLS, mesh.CellType.quadrilateral),
    mesh.create_unit_cube(MPI.COMM_WORLD, NCELLS, NCELLS, NCELLS, mesh.CellType.hexahedron),
    ])
def test(domain):
    x = domain.geometry.x
    tdim = domain.topology.dim

    xaxis = line_at([0.0, 0.0], ["y", "z"])
    yaxis = line_at([0.0, 0.0], ["x", "z"])
    zaxis = line_at([0.0, 0.0], ["x", "y"])

    axes = (xaxis, yaxis, zaxis)
    expected = {2: (NCELLS + 1, NCELLS + 1, 1), 3: (NCELLS + 1, ) * 3}
    for k, axis in enumerate(axes):
        assert np.isclose(np.sum(axis(x.T)), expected[tdim][k])
