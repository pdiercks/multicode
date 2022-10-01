import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from multi.bcs import BoundaryConditions

"""Note: topological vs. geometrical

It seems that `locate_dofs_geometrical` does not work with V.sub
since at some point the dof coordinates need to be tabulated
which is not possible for a subspace.
However, one could always first locate the entities geometrically
if this is more convenient.

```python
from dolfinx.fem import dirichletbc
from dolfinx.mesh import locate_entities_boundary, locate_dofs_topological

def plane_at(coordinate, dim):

    def boundary(x):
        return np.isclose(x[dim], coordinate)

    return boundary

bottom = plane_at(0., 1)

bottom_boundary_facets = locate_entities_boundary(
    domain, domain.topology.dim - 1, bottom
)
bottom_boundary_dofs_y = locate_dofs_topological(
    V.sub(1), domain.topology.dim - 1, bottom_boundary_facets
)
fix_uy = dirichletbc(ScalarType(0), bottom_boundary_dofs_y, V.sub(1))
```

"""


# FIXME fails in parallel
# need to understand how dof_indices (array) are distributed
def test_vector_geom():
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, dolfinx.mesh.CellType.quadrilateral
    )
    V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 2))

    bc_handler = BoundaryConditions(domain, V)

    def left(x):
        return np.isclose(x[0], 0.0)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # entire boundary; should have 64 * 2 dofs
    # constrain entire boundary only for the x-component
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    bc_handler.add_dirichlet_bc(
        ScalarType(0), boundary_facets, sub=0, method="topological", entity_dim=fdim
    )
    # constrain left boundary as well
    # FIXME does V.sub work really only with method="topological"??
    zero = np.array([0.0, 0.0], dtype=ScalarType)
    bc_handler.add_dirichlet_bc(zero, left, method="geometrical")

    bcs = bc_handler.bcs
    ndofs = 0
    for bc in bcs:
        ndofs += bc.dof_indices()[1]

    assert ndofs == 64 + 34


def test_scalar_geom():
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.FunctionSpace(domain, ("CG", 2))

    bc_handler = BoundaryConditions(domain, V)

    def left(x):
        return np.isclose(x[0], 0.0)

    bc_handler.add_dirichlet_bc(ScalarType(0), left, method="geometrical")

    bcs = bc_handler.bcs
    my_bc = bcs[0]

    ndofs = my_bc.dof_indices()[1]
    all_ndofs = domain.comm.allreduce(ndofs, op=MPI.SUM)
    assert all_ndofs == 17
    assert my_bc.g.value == 0.0


def test_scalar_topo():
    n = 20
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    V = dolfinx.fem.FunctionSpace(domain, ("CG", 2))

    bc_handler = BoundaryConditions(domain, V)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # entire boundary; should have (n+1+n)*4 - 4 = 8n dofs
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    bc_handler.add_dirichlet_bc(ScalarType(0), boundary_facets, entity_dim=fdim)

    bcs = bc_handler.bcs
    my_bc = bcs[0]

    ndofs = my_bc.dof_indices()[1]
    all_ndofs = domain.comm.allreduce(ndofs, op=MPI.SUM)
    assert all_ndofs == 8 * n
    assert my_bc.g.value == 0.0


if __name__ == "__main__":
    test_scalar_geom()
    test_scalar_topo()
    test_vector_geom()
