from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type
from basix.ufl import element
import numpy as np
from multi.bcs import BoundaryConditions
from multi.interpolation import interpolate


def test_FunctionSpace():
    coarse = mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    fine = mesh.create_unit_interval(MPI.COMM_WORLD, 20)

    line1 = element("Lagrange", coarse.basix_cell(), 1, shape=())
    line2 = element("Lagrange", fine.basix_cell(), 2, shape=())
    W = fem.functionspace(coarse, line1)
    V = fem.functionspace(fine, line2)

    w = fem.Function(W)
    w.interpolate(lambda x: x[0] * 12.0)

    # get points of the fine grid and evaluate w at points
    points = V.tabulate_dof_coordinates()
    # transpose points although correct shape
    # interpolate should handle this
    values = interpolate(w, points.T)

    # compare against u in V
    u = fem.Function(V)
    u.interpolate(lambda x: x[0] * 12.0)

    assert np.allclose(u.x.array, values.reshape(u.x.array.shape))


def test_VectorFunctionSpace():
    coarse = mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    fine = mesh.create_unit_interval(MPI.COMM_WORLD, 20)

    line1 = element("Lagrange", coarse.basix_cell(), 1, shape=(2,))
    line2 = element("Lagrange", fine.basix_cell(), 2, shape=(2,))
    W = fem.functionspace(coarse, line1)
    V = fem.functionspace(fine, line2)

    w = fem.Function(W)
    w.interpolate(lambda x: np.array([4.2 * x[0], 2.7 * x[0]]))

    # get points of the fine grid and evaluate w at points
    points = V.tabulate_dof_coordinates()
    values = interpolate(w, points)

    # compare against u in V
    u = fem.Function(V)
    u.interpolate(lambda x: np.array([4.2 * x[0], 2.7 * x[0]]))

    assert np.allclose(u.x.array, values.reshape(u.x.array.shape))


def test_VectorFunctionSpace_square():
    coarse = mesh.create_unit_square(
        MPI.COMM_WORLD, 3, 3, mesh.CellType.quadrilateral
    )
    fine = mesh.create_unit_square(
        MPI.COMM_WORLD, 30, 30, mesh.CellType.triangle
    )

    quad1 = element("Lagrange", coarse.basix_cell(), 1, shape=(2,))
    quad2 = element("Lagrange", fine.basix_cell(), 2, shape=(2,))
    W = fem.functionspace(coarse, quad1)
    V = fem.functionspace(fine, quad2)

    w = fem.Function(W)
    w.interpolate(lambda x: np.array([4.2 * x[0], 2.7 * x[1]]))

    # get points of the fine grid and evaluate w at points
    points = V.tabulate_dof_coordinates()
    values = interpolate(w, points)

    # compare against u in V
    u = fem.Function(V)
    u.interpolate(lambda x: np.array([4.2 * x[0], 2.7 * x[1]]))

    assert np.allclose(u.x.array, values.reshape(u.x.array.shape))


def test_VectorFunctionSpace_square_Boundary():
    coarse = mesh.create_unit_square(
        MPI.COMM_WORLD, 3, 3, mesh.CellType.quadrilateral
    )
    fine = mesh.create_unit_square(
        MPI.COMM_WORLD, 30, 30, mesh.CellType.triangle
    )

    quad1 = element("Lagrange", coarse.basix_cell(), 1, shape=(2,))
    W = fem.functionspace(coarse, quad1)
    quad2 = element("Lagrange", fine.basix_cell(), 2, shape=(2,))
    V = fem.functionspace(fine, quad2)

    w = fem.Function(W)
    w.interpolate(lambda x: np.array([4.2 * x[0], 2.7 * x[1]]))

    # get points of the fine grid of the boundary and evaluate w at points
    bc_handler = BoundaryConditions(fine, V)

    tdim = fine.topology.dim
    fdim = tdim - 1
    fine.topology.create_connectivity(fdim, tdim)

    # entire boundary; should have 240 * 2 dofs in total
    # each node has 2 dofs, but only need the coordinate once for interpolation
    boundary_facets = mesh.exterior_facet_indices(fine.topology)
    bc_handler.add_dirichlet_bc(
        default_scalar_type(0), boundary_facets, sub=0, method="topological", entity_dim=fdim
    )
    bcs = bc_handler.bcs
    dofs, num_dofs = bcs[0]._cpp_object.dof_indices()
    assert num_dofs == 240

    def xdofs_VectorFunctionSpace(V):
        bs = V.dofmap.bs
        x = V.tabulate_dof_coordinates()
        x_dofs = np.repeat(x, repeats=bs, axis=0)
        return x_dofs

    # dof indices are in range [0, Vdim]
    # therefore, need to repeat the dof coordinates returned by V.tabulate_dof_coordinates()
    points = xdofs_VectorFunctionSpace(V)[dofs]
    w_values = interpolate(w, points)

    # compare against u in V
    u = fem.Function(V)
    u.interpolate(lambda x: np.array([4.2 * x[0], 2.7 * x[1]]))

    bc_handler.clear()
    zero = np.array([0.0, 0.0], dtype=default_scalar_type)
    bc_handler.add_dirichlet_bc(
        zero, boundary_facets, method="topological", entity_dim=fdim
    )
    bcs = bc_handler.bcs
    dofs, num_dofs = bcs[0]._cpp_object.dof_indices()
    assert num_dofs == 240 * 2
    u_values = u.x.array[dofs]

    assert np.allclose(u_values, w_values.flatten())


if __name__ == "__main__":
    test_FunctionSpace()
    test_VectorFunctionSpace()
    test_VectorFunctionSpace_square()
    test_VectorFunctionSpace_square_Boundary()
