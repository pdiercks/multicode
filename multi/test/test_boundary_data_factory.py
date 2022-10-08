import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from multi.bcs import BoundaryDataFactory


def test():
    n = 8
    rectangle = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [[-1.0, -1.0], [1.0, 1.0]], [n, n]
    )  # [-1, 1]x[-1, 1]
    interval = dolfinx.mesh.create_interval(
        MPI.COMM_WORLD, n, [-1.0, 1.0]
    )  # [-1, 1]x{0}

    degree = 1
    V = dolfinx.fem.FunctionSpace(rectangle, ("CG", degree))
    W = dolfinx.fem.FunctionSpace(interval, ("CG", degree))

    # generate mode on interval (bottom edge)
    w = dolfinx.fem.Function(W)
    w.interpolate(lambda x: x[0] * (1.0 - x[0] ** 2))
    mode = w.x.array

    def bottom(x):
        return np.isclose(x[1], -1.0)

    from IPython import embed

    embed()

    data_factory = BoundaryDataFactory(rectangle, V)
    # FIXME find out how to create dirichletbc with values being a numpy array
    data_factory.set_values_via_bc(mode, boundary=bottom, method="geometrical")
    bc = data_factory.create_bc()

    u = dolfinx.fem.Function(V)
    uvec = u.vector
    uvec.zeroEntries()
    dolfinx.fem.petsc.set_bc(uvec, [bc], scale=1.0)
    uvec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
    )

    bottom_dofs = dolfinx.fem.locate_dofs_geometrical(V, bottom)
    assert np.allclose(uvec[bottom_dofs], mode)


if __name__ == "__main__":
    test()
