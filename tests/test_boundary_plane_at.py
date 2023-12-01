import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import mesh
from multi.boundary import plane_at

NCELLS = 4


@pytest.mark.parametrize("domain",[
    mesh.create_unit_square(MPI.COMM_WORLD, NCELLS, NCELLS, mesh.CellType.quadrilateral),
    mesh.create_unit_cube(MPI.COMM_WORLD, NCELLS, NCELLS, NCELLS, mesh.CellType.hexahedron),
    ])
def test(domain):
    x = domain.geometry.x
    tdim = domain.topology.dim
    fdim = tdim - 1

    bottom = plane_at(0.0, "y")
    top = plane_at(1.0, "y")
    left = plane_at(0.0, "x")
    right = plane_at(1.0, "x")
    front = plane_at(0.0, "z")
    back = plane_at(1.0, "z")

    boundaries = [bottom, right, top, left]

    if fdim == 1:
        for edge in boundaries:
            assert np.isclose(np.sum(edge(x.T)), NCELLS + 1)
    elif fdim == 2:
        boundaries += [front, back]
        for plane in boundaries:
            assert np.isclose(np.sum(plane(x.T)), (NCELLS + 1) ** 2)
