from mpi4py import MPI
import dolfinx
import numpy as np
from basix.ufl import element
from petsc4py import PETSc
from multi.boundary import plane_at


def test():
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, dolfinx.mesh.CellType.quadrilateral
    )
    quad = element("Lagrange", domain.basix_cell(), 1, shape=())
    V = dolfinx.fem.FunctionSpace(domain, quad)

    bottom = plane_at(0.0, "y")
    top = plane_at(1.0, "y")
    left = plane_at(0.0, "x")
    right = plane_at(1.0, "x")

    scalar_type = dolfinx.default_scalar_type

    dofs = dolfinx.fem.locate_dofs_geometrical(V, bottom)
    bc = dolfinx.fem.dirichletbc(scalar_type(42), dofs, V)
    ndofs = bc._cpp_object.dof_indices()[1]
    assert ndofs == 9
    assert bc.g.value == 42

    def on_boundary(x):
        return np.logical_or(top(x), bottom(x))

    dofs = dolfinx.fem.locate_dofs_geometrical(V, on_boundary)
    bc = dolfinx.fem.dirichletbc(scalar_type(17), dofs, V)
    ndofs = bc._cpp_object.dof_indices()[1]
    assert ndofs == 9 * 2
    assert bc.g.value == 17

    def origin(x):
        return np.logical_and(left(x), bottom(x))

    dofs = dolfinx.fem.locate_dofs_geometrical(V, origin)
    bc = dolfinx.fem.dirichletbc(scalar_type(21), dofs, V)
    ndofs = bc._cpp_object.dof_indices()[1]
    assert ndofs == 1
    assert bc.g.value == 21

    def l_shaped_boundary(x):
        return np.logical_or(top(x), right(x))

    facet_dim = 1
    boundary_facets = dolfinx.mesh.locate_entities_boundary(
        domain, facet_dim, l_shaped_boundary
    )
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, facet_dim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(scalar_type(666), boundary_dofs, V)
    ndofs = bc._cpp_object.dof_indices()[1]
    assert ndofs == 17
    assert bc.g.value == 666


if __name__ == "__main__":
    test()
