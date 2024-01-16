from mpi4py import MPI
import dolfinx
import numpy as np
from basix.ufl import element
from petsc4py import PETSc
from dolfinx.fem.petsc import set_bc
from multi.bcs import BoundaryDataFactory


def test():
    n = 8
    rectangle = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, np.array([[-1.0, -1.0], [1.0, 1.0]]), [n, n]
    )  # [-1, 1]x[-1, 1]
    interval = dolfinx.mesh.create_interval(
        MPI.COMM_WORLD, n, [-1.0, 1.0]
    )  # [-1, 1]x{0}

    degree = 1
    quad = element("Lagrange", rectangle.basix_cell(), degree, shape=())
    line = element("Lagrange", interval.basix_cell(), degree, shape=())
    V = dolfinx.fem.FunctionSpace(rectangle, quad)
    W = dolfinx.fem.FunctionSpace(interval, line)

    # generate mode on interval (bottom edge)
    w = dolfinx.fem.Function(W)
    w.interpolate(lambda x: x[0] * (1.0 - x[0] ** 2))
    mode = w.x.array

    def bottom(x):
        return np.isclose(x[1], -1.0)

    bottom_dofs = dolfinx.fem.locate_dofs_geometrical(V, bottom)

    data_factory = BoundaryDataFactory(rectangle, V)
    f = data_factory.create_function_values(mode, bottom_dofs)
    bc_0 = data_factory.create_bc(f)
    assert np.allclose(bc_0._cpp_object.dof_indices()[0], data_factory.boundary_dofs)
    g = data_factory.create_function_values(
        np.ones(bottom_dofs.size, dtype=np.float64), bottom_dofs
    )
    bc_1 = data_factory.create_bc(g)

    u = dolfinx.fem.Function(V)
    uvec = u.vector
    uvec.zeroEntries()
    set_bc(uvec, [bc_0], scale=1.0)
    uvec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
    )
    assert np.allclose(uvec[bottom_dofs], mode)
    uvec.zeroEntries()
    set_bc(uvec, [bc_1], scale=1.0)
    uvec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
    )
    assert np.allclose(uvec[bottom_dofs], np.ones(bottom_dofs.size))

    my_bc = {"value": PETSc.ScalarType(1.), "boundary": bottom, "method": "geometrical"}
    f = data_factory.create_function_bc([my_bc])
    bc_f = data_factory.create_bc(f)
    assert np.isclose(np.sum(bc_f.g.x.array), 9)


if __name__ == "__main__":
    test()
