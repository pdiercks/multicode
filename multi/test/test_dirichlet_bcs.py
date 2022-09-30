import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from multi.bcs import BoundaryConditions


def test_scalar_geom():
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.FunctionSpace(domain, ("CG", 2))

    bc_handler = BoundaryConditions(domain, V)

    def left(x):
        return np.isclose(x[0], 0.0)

    bc_handler.add_dirichlet(ScalarType(0), left, method="geometrical")

    bcs = bc_handler.bcs()
    my_bc = bcs[0]
    assert my_bc.dof_indices()[1] == 17
    assert my_bc.g.value == 0.


def test_scalar_topo():
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.FunctionSpace(domain, ("CG", 2))

    bc_handler = BoundaryConditions(domain, V)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # entire boundary; should have 64 dofs
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    bc_handler.add_dirichlet(ScalarType(0), boundary_facets, entity_dim=fdim)

    bcs = bc_handler.bcs()
    my_bc = bcs[0]
    assert my_bc.dof_indices()[1] == 64
    assert my_bc.g.value == 0.


if __name__ == "__main__":
    test_scalar_geom()
    test_scalar_topo()
