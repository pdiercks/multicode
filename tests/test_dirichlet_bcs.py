from mpi4py import MPI
import dolfinx
from basix.ufl import element
import numpy as np
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
fix_uy = dirichletbc(dolfinx.default_scalar_type(0), bottom_boundary_dofs_y, V.sub(1))
```

"""


def test_vector_geom():
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, dolfinx.mesh.CellType.quadrilateral
    )
    ve = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    V = dolfinx.fem.functionspace(domain, ve)

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
        dolfinx.default_scalar_type(0), boundary_facets, sub=0, method="topological", entity_dim=fdim
    )
    # constrain left boundary as well
    zero = np.array([0.0, 0.0], dtype=dolfinx.default_scalar_type)
    bc_handler.add_dirichlet_bc(zero, left, method="geometrical")

    bcs = bc_handler.bcs
    ndofs = 0
    for bc in bcs:
        ndofs += bc._cpp_object.dof_indices()[1]

    assert ndofs == 64 + 34


def test_vector_geom_component_wise():
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, dolfinx.mesh.CellType.quadrilateral
    )
    ve = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    V = dolfinx.fem.functionspace(domain, ve)

    bc_handler = BoundaryConditions(domain, V)

    def left(x):
        return np.isclose(x[0], 0.0)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    zero = dolfinx.default_scalar_type(0.)
    bc_handler.add_dirichlet_bc(zero, left, method="geometrical", sub=0, entity_dim=fdim)

    bcs = bc_handler.bcs
    ndofs = 0
    for bc in bcs:
        ndofs += bc._cpp_object.dof_indices()[1]

    assert ndofs == 17


def test_scalar_geom():
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    ve = element("Lagrange", domain.basix_cell(), 2, shape=())
    V = dolfinx.fem.functionspace(domain, ve)

    bc_handler = BoundaryConditions(domain, V)

    def left(x):
        return np.isclose(x[0], 0.0)

    bc_handler.add_dirichlet_bc(dolfinx.default_scalar_type(0), left, method="geometrical")

    bcs = bc_handler.bcs
    my_bc = bcs[0]

    ndofs = my_bc._cpp_object.dof_indices()[1]
    all_ndofs = domain.comm.allreduce(ndofs, op=MPI.SUM)
    assert all_ndofs == 17
    assert my_bc.g.value == 0.0


def test_scalar_topo():
    n = 20
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    ve = element("Lagrange", domain.basix_cell(), 2, shape=())
    V = dolfinx.fem.functionspace(domain, ve)

    bc_handler = BoundaryConditions(domain, V)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # entire boundary; should have (n+1+n)*4 - 4 = 8n dofs
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    bc_handler.add_dirichlet_bc(dolfinx.default_scalar_type(0), boundary_facets, entity_dim=fdim)

    bcs = bc_handler.bcs
    my_bc = bcs[0]

    ndofs = my_bc._cpp_object.dof_indices()[1]
    all_ndofs = domain.comm.allreduce(ndofs, op=MPI.SUM)
    assert all_ndofs == 8 * n
    assert my_bc.g.value == 0.0


if __name__ == "__main__":
    test_scalar_geom()
    test_scalar_topo()
    test_vector_geom()
    test_vector_geom_component_wise()
