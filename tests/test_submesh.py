from mpi4py import MPI
from dolfinx import mesh
from multi.boundary import within_range, plane_at
import pytest

@pytest.mark.parametrize("edim",[1, 2])
def test(edim):
    nx = ny = 40
    cell_type = mesh.CellType.quadrilateral
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type)
    tdim = domain.topology.dim

    entities = None
    num_cells = domain.topology.index_map(tdim).size_global
    if edim == 1:
        bottom = plane_at(0., "y")
        entities = mesh.locate_entities_boundary(domain, edim, bottom)
        cell_type = mesh.CellType.interval
        num_cells = nx
    elif edim == 2:
        omega_in = within_range([0.3, 0.3], [0.6, 0.6])
        entities = mesh.locate_entities(domain, edim, omega_in)
        num_cells = int((nx * (0.6 - 0.3)) ** 2)
    
    # submesh, entity_map, vertex_map, geom_map = mesh.create_submesh(domain, edim, entities)
    submesh, _, _, _ = mesh.create_submesh(domain, edim, entities)
    assert isinstance(submesh, mesh.Mesh)
    assert submesh.topology.cell_types[0] == cell_type
    assert submesh.topology.index_map(edim).size_global == num_cells


if __name__ == "__main__":
    test(1)
    test(2)
