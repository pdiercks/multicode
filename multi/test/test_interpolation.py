import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from multi.bcs import BoundaryConditions


def interpolate(u, points):
    """evaluate u at points"""
    V = u.function_space
    domain = V.mesh
    bb_tree = dolfinx.geometry.BoundingBoxTree(domain, domain.topology.dim)

    cells = []
    points_on_proc = []

    if not points.shape[1] == 3:
        points = points.T
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        domain, cell_candidates, points
    )
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc, cells)
    return u_values


def test_FunctionSpace():
    coarse = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    fine = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 20)

    V = dolfinx.fem.FunctionSpace(fine, ("CG", 2))
    W = dolfinx.fem.FunctionSpace(coarse, ("CG", 1))

    w = dolfinx.fem.Function(W)
    w.interpolate(lambda x: x[0] * 12.)

    # get points of the fine grid and evaluate w at points
    points = V.tabulate_dof_coordinates()
    values = interpolate(w, points)

    # compare against u in V
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] * 12.)

    assert np.allclose(u.x.array, values.reshape(u.x.array.shape))


def test_VectorFunctionSpace():
    coarse = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    fine = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 20)

    V = dolfinx.fem.VectorFunctionSpace(fine, ("CG", 2), dim=2)
    W = dolfinx.fem.VectorFunctionSpace(coarse, ("CG", 1), dim=2)

    w = dolfinx.fem.Function(W)
    w.interpolate(lambda x: np.array([4.2 * x[0], 2.7 * x[0]]))


    # get points of the fine grid and evaluate w at points
    points = V.tabulate_dof_coordinates()
    values = interpolate(w, points)

    # compare against u in V
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.array([4.2 * x[0], 2.7 * x[0]]))

    assert np.allclose(u.x.array, values.reshape(u.x.array.shape))


def test_VectorFunctionSpace_square():
    coarse = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, dolfinx.mesh.CellType.quadrilateral)
    fine = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 30, 30, dolfinx.mesh.CellType.triangle)

    V = dolfinx.fem.VectorFunctionSpace(fine, ("CG", 2), dim=2)
    W = dolfinx.fem.VectorFunctionSpace(coarse, ("CG", 1), dim=2)

    w = dolfinx.fem.Function(W)
    w.interpolate(lambda x: np.array([4.2 * x[0], 2.7 * x[1]]))


    # get points of the fine grid and evaluate w at points
    points = V.tabulate_dof_coordinates()
    values = interpolate(w, points)

    # compare against u in V
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.array([4.2 * x[0], 2.7 * x[1]]))

    assert np.allclose(u.x.array, values.reshape(u.x.array.shape))


def test_VectorFunctionSpace_square_Boundary():
    coarse = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, dolfinx.mesh.CellType.quadrilateral)
    fine = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 30, 30, dolfinx.mesh.CellType.triangle)

    V = dolfinx.fem.VectorFunctionSpace(fine, ("CG", 2), dim=2)
    W = dolfinx.fem.VectorFunctionSpace(coarse, ("CG", 1), dim=2)

    w = dolfinx.fem.Function(W)
    w.interpolate(lambda x: np.array([4.2 * x[0], 2.7 * x[1]]))

    # get points of the fine grid of the boundary and evaluate w at points
    bc_handler = BoundaryConditions(fine, V)

    tdim = fine.topology.dim
    fdim = tdim - 1
    fine.topology.create_connectivity(fdim, tdim)

    # entire boundary; should have 240 * 2 dofs in total
    # each node has 2 dofs, but only need the coordinate once for interpolation
    boundary_facets = dolfinx.mesh.exterior_facet_indices(fine.topology)
    bc_handler.add_dirichlet_bc(
        ScalarType(0), boundary_facets, sub=0, method="topological", entity_dim=fdim
    )
    bcs = bc_handler.bcs
    dofs, num_dofs = bcs[0].dof_indices()
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
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.array([4.2 * x[0], 2.7 * x[1]]))

    bc_handler.clear()
    zero = np.array([0., 0.], dtype=ScalarType)
    bc_handler.add_dirichlet_bc(
        zero, boundary_facets, method="topological", entity_dim=fdim
    )
    bcs = bc_handler.bcs
    dofs, num_dofs = bcs[0].dof_indices()
    assert num_dofs == 240 * 2
    u_values = u.x.array[dofs]

    assert np.allclose(u_values, w_values.flatten())


if __name__ == "__main__":
    test_FunctionSpace()
    test_VectorFunctionSpace()
    test_VectorFunctionSpace_square()
    test_VectorFunctionSpace_square_Boundary()
