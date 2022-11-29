import dolfinx
from mpi4py import MPI
import numpy as np
from multi.projection import fine_scale_part
from multi.boundary import point_at


def test_interval():
    Ω = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 1)
    ω = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 30)

    W = dolfinx.fem.FunctionSpace(Ω, ("P", 1))
    V = dolfinx.fem.FunctionSpace(ω, ("P", 2))

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] ** 2 - x[0] + 1./4)

    f = fine_scale_part(u, W)

    ref = dolfinx.fem.Function(V)
    ref.interpolate(lambda x: x[0] ** 2 - x[0])

    np.testing.assert_allclose(f.x.array, ref.x.array, atol=1e-8)


def test_square():
    Ω = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    ω = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 30, 30)

    W = dolfinx.fem.VectorFunctionSpace(Ω, ("P", 1))
    V = dolfinx.fem.VectorFunctionSpace(ω, ("P", 2))

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: (x[0] ** 2, x[1] ** 2))

    f = fine_scale_part(u, W)

    points = np.array([
        [0.0, 0.0, 0.],
        [0.5, 0.0, 0.],
        [1.0, 0.0, 0.],
        [0.0, 0.5, 0.],
        [0.5, 0.5, 0.],
        [1.0, 0.5, 0.],
        [0.0, 1.0, 0.],
        [0.5, 1.0, 0.],
        [1.0, 1.0, 0.],])

    for x in points:
        dd = dolfinx.fem.locate_dofs_geometrical(V, point_at(x))
        bc = dolfinx.fem.dirichletbc(np.array([0, 0], dtype=float), dd, V)
        dofs = bc.dof_indices()[0]

        np.testing.assert_allclose(f.x.array[dofs], np.zeros(dofs.size), atol=1e-12)

    assert np.sum(np.abs(f.x.array[:])) > 0.


if __name__ == "__main__":
    test_interval()
    test_square()
