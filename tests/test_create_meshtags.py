"""test cell and facet tag creation"""

import pytest
from mpi4py import MPI
from dolfinx import mesh
from multi.boundary import within_range, plane_at
from multi.preprocessing import create_meshtags


@pytest.mark.parametrize("domain", [
    mesh.create_unit_square(MPI.COMM_SELF, 6, 6, mesh.CellType.quadrilateral),
    mesh.create_unit_cube(MPI.COMM_SELF, 6, 6, 6, mesh.CellType.hexahedron),
    ])
def test_cells(domain):
    tdim = domain.topology.dim

    marker_id = int(39)
    subdomain = {
            "inner": (marker_id, within_range([0., 0., 0.], [1./3, 1./3, 1./3]))
            }

    tags, marked = create_meshtags(domain, tdim, subdomain)
    assert marked["inner"] == 39
    assert tags.find(39).size > 0
    if tdim == 2:
        assert tags.find(39).size == 4


@pytest.mark.parametrize("domain", [
    mesh.create_unit_square(MPI.COMM_SELF, 6, 6, mesh.CellType.quadrilateral),
    mesh.create_unit_cube(MPI.COMM_SELF, 6, 6, 6, mesh.CellType.hexahedron),
    ])
def test_facets(domain):
    tdim = domain.topology.dim

    boundaries = {
            "left": (int(4), plane_at(0.0, "x")),
            "right": (int(112), plane_at(1.0, "x"))
            }

    tags, marked = create_meshtags(domain, tdim-1, boundaries)
    assert marked["left"] == 4
    assert marked["right"] == 112
    assert tags.find(4).size > 0
    assert tags.find(112).size > 0
    assert tags.find(4).size == tags.find(112).size

    boundaries.clear()
    boundaries.update({"top": (int(42), plane_at(1.0, "y"))})

    tags_, marked_ = create_meshtags(domain, tdim-1, boundaries, tags=tags)
    assert marked_["top"] == 42
    assert tags_.find(4).size > 0
    assert tags_.find(112).size > 0
    assert tags_.find(42).size > 0
    assert tags_.find(4).size == tags_.find(112).size

    with pytest.raises(NotImplementedError):
        bndry = {"left": (int(44), plane_at(0.0, "x"))}
        create_meshtags(domain, tdim-1, bndry, tags=tags_)
