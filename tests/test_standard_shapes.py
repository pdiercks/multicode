"""test standard shape functions"""

import pytest
from mpi4py import MPI
from dolfinx import fem, mesh
from basix.ufl import element
import numpy as np
from multi.shapes import NumpyLine, NumpyQuad


@pytest.mark.parametrize("value_shape",[(),(2,)])
def test_numpy_line(value_shape):
    interval = mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    fe = element("P", interval.basix_cell(), 2, shape=value_shape)
    V = fem.functionspace(interval, fe)

    line2 = NumpyLine(np.array([0, 1]))
    line3 = NumpyLine(np.array([0, 1, 0.5]))
    with pytest.raises(NotImplementedError):
        NumpyLine(np.array([0, 1, 0.25, 0.5]))

    bs = V.dofmap.bs
    x = V.tabulate_dof_coordinates()
    x_dofs = np.repeat(x, repeats=bs, axis=0)
    n_verts = len(x_dofs)

    shapes = line2.interpolate(V, 0)
    assert np.isclose(np.sum(shapes), n_verts)
    shapes = line3.interpolate(V, 0)
    assert np.isclose(np.sum(shapes), n_verts)


def test_numpy_quad():
    square = mesh.create_unit_square(
        MPI.COMM_WORLD, 20, 20, mesh.CellType.quadrilateral
    )
    ve = element("P", square.basix_cell(), 2, shape=(2,))
    V = fem.functionspace(square, ve)

    g = fem.Function(V)
    g.interpolate(lambda x: ((1.0 - x[0]) * (1.0 - x[1]), np.zeros_like(x[0])))

    quad4 = NumpyQuad(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    shapes4 = quad4.interpolate(V)
    assert np.allclose(shapes4[0], g.vector[:])
    assert np.isclose(np.sum(shapes4), 2 * len(V.tabulate_dof_coordinates()))

    quad8 = NumpyQuad(
        np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]]
        )
    )
    shapes8 = quad8.interpolate(V)
    assert np.isclose(np.sum(shapes8), 2 * len(V.tabulate_dof_coordinates()))

    quad9 = NumpyQuad(
        np.array(
            [
                [-1, -1],
                [1, -1],
                [1, 1],
                [-1, 1],
                [0, -1],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, 0],
            ]
        )
    )
    rectangle = mesh.create_rectangle(
        MPI.COMM_WORLD, np.array([[0.0, 0.0], [2.0, 2.0]]), [8, 8]
    )
    xg = rectangle.geometry.x
    xg += np.array([-1, -1, 0], dtype=np.float64)

    fe = element("P", rectangle.basix_cell(), 2, shape=())
    V = fem.functionspace(rectangle, fe)
    x_dofs = V.tabulate_dof_coordinates()
    x = x_dofs[:, 0]
    y = x_dofs[:, 1]

    def analytic(x, y):
        return x * y * (x - 1) * (y - 1) / 4

    shapes9 = quad9.interpolate(V)
    u = analytic(x, y)
    e = u - shapes9[0]
    assert np.linalg.norm(e) < 1e-14
