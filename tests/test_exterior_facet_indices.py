import pytest
import numpy as np
from dolfinx.mesh import create_unit_interval, create_unit_square, exterior_facet_indices, create_submesh, locate_entities_boundary
from mpi4py.MPI import COMM_WORLD as comm


def get_boundary_submesh():
    square = create_unit_square(comm, 10, 10)
    tdim = square.topology.dim
    fdim = tdim - 1

    def bottom(x):
        return np.isclose(x[1], 0.0)

    bottom_facets = locate_entities_boundary(square, fdim, bottom)
    domain = create_submesh(square, fdim, bottom_facets)[0]
    # NOTE see cpp/dolfinx/mesh/Mesh.h, line 140
    # create_submesh returns
    # (dolfinx.mesh.Mesh, entities, vertices, geometry)
    # such that submesh.geometry.x == parent.geometry.x[vertices]
    # entities are the entities from which the submesh was created
    # (i.e. bottom_facets in the above example)
    return domain


@pytest.mark.parametrize("domain", [
    create_unit_interval(comm, 10),
    create_unit_square(comm, 10, 10),
    get_boundary_submesh()
    ])
def test(domain):
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = exterior_facet_indices(domain.topology)
    assert len(boundary_facets) > 0
