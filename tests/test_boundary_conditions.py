from mpi4py import MPI
import pytest
from dolfinx import fem, mesh, default_scalar_type
from basix.ufl import element
import numpy as np
from multi.boundary import point_at, plane_at
from multi.bcs import BoundaryConditions
from multi.preprocessing import create_meshtags

# cases
# V has shape (gdim,) for gdim in 1, 2, 3
# method topological, geometrical
# topological: boundary has to be int or entities
# geometrical: boundary is Callable

# append existing bc
# has_dirichlet



def count_dofs(domain: mesh.Mesh, bcs: list[fem.DirichletBC], expected_value):
    ndofs = 0
    for bc in bcs:
        dofs_on_proc = bc._cpp_object.dof_indices()[1]
        ndofs += domain.comm.allreduce(dofs_on_proc, op=MPI.SUM)
    assert np.isclose(ndofs, expected_value)


def test_scalar():
    """test scalar function space"""
    nx = ny = 8
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
    ve = element("Lagrange", domain.basix_cell(), 2, shape=())
    V = fem.functionspace(domain, ve)

    bc_handler = BoundaryConditions(domain, V)
    assert not bc_handler.has_dirichlet

    with pytest.raises(AttributeError):
        # there are no facet tags defined
        bc_handler.add_dirichlet_bc(default_scalar_type(0.), int(2), entity_dim=1)

    # ### method="geometrical"

    def left(x):
        return np.isclose(x[0], 0.0)

    bc_handler.add_dirichlet_bc(default_scalar_type(0), left, method="geometrical")

    ndofs = (ny + 1) + ny # vertex dofs + mid point dofs
    count_dofs(domain, bc_handler.bcs, ndofs)

    bc_handler.bcs.clear()
    assert not bc_handler.has_dirichlet

    # ### method="topological"
    # using boundary_facets: np.ndarray

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    # entire boundary; should have (nx+1+nx)*4 - 4 = 8nx dofs
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    bc_handler.add_dirichlet_bc(default_scalar_type(0), boundary_facets, entity_dim=fdim)

    ndofs = 8 * nx
    count_dofs(domain, bc_handler.bcs, ndofs)


def test_vector():
    """test vector-valued function space"""
    nx = ny = 8
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral
    )
    ve = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
    V = fem.functionspace(domain, ve)

    tdim = domain.topology.dim
    fdim = tdim - 1
    bottom_tag = int(12)
    top_tag = int(47)
    boundaries = {
            "bottom": (bottom_tag, plane_at(0., "y")),
            "top": (top_tag, plane_at(1., "y")),
            }
    facet_tags, marked = create_meshtags(domain, fdim, boundaries)
    assert marked["bottom"] == bottom_tag
    assert marked["top"] == top_tag

    bc_handler = BoundaryConditions(domain, V, facet_tags=facet_tags)

    # ### geometrical
    origin = point_at([0., 0.])
    zero = fem.Constant(domain, (default_scalar_type(0.), ) * 2)
    bc_handler.add_dirichlet_bc(zero, origin, method="geometrical") # type: ignore

    # ### geometrical, sub=0
    bottom_right = point_at([1., 0.])
    bc_handler.add_dirichlet_bc(default_scalar_type(0.), bottom_right, method="geometrical", sub=0, entity_dim=0)

    bcs = bc_handler.bcs
    count_dofs(domain, bcs, 3)

    # ### topological, meshtags
    # value: Function, some expression
    bc_handler.clear()
    f = fem.Function(V)
    bc_handler.add_dirichlet_bc(f, bottom_tag, entity_dim=fdim)

    expr = lambda x: (x[0], x[1])
    bc_handler.add_dirichlet_bc(expr, top_tag, entity_dim=fdim)

    with pytest.raises(ValueError):
        # try undefined tag value
        bc_handler.add_dirichlet_bc(expr, int(99), entity_dim=fdim)

    ndofs = (nx +1) + nx
    n_boundaries = len(boundaries)
    n_comp = ve.value_shape()[0]
    count_dofs(domain, bc_handler.bcs, ndofs * n_boundaries * n_comp)

    with pytest.raises(ValueError):
        # try undefined tag value for neumann bc
        bc_handler.add_neumann_bc(int(1012), zero)


if __name__ == "__main__":
    test_scalar()
    test_vector()
