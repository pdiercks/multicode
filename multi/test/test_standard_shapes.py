"""test standard shape functions"""
import dolfin as df
import numpy as np
from multi.shapes import NumpyLine, NumpyQuad, mapping


def test():
    mesh = df.UnitIntervalMesh(10)
    V = df.FunctionSpace(mesh, "CG", 2)

    line2 = NumpyLine(np.array([0, 1]))
    line3 = NumpyLine(np.array([0, 1, 0.5]))
    x_dofs = V.tabulate_dof_coordinates()
    n_verts = len(x_dofs)

    shapes = line2.interpolate(x_dofs, (1,))
    assert np.isclose(np.sum(shapes), n_verts)
    shapes = line3.interpolate(x_dofs, (1,))
    assert np.isclose(np.sum(shapes), n_verts)

    mesh = df.UnitSquareMesh(20, 20, "crossed")
    V = df.VectorFunctionSpace(mesh, "CG", 2)

    gexpr = df.Expression(("(1 - x[0]) * (1 - x[1])", "0"), degree=1)
    g = df.interpolate(gexpr, V)

    quad4 = NumpyQuad(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    shapes4 = quad4.interpolate(V.sub(0).collapse().tabulate_dof_coordinates(), (2,))
    assert np.allclose(shapes4[0], g.vector()[:])
    assert np.isclose(np.sum(shapes4), len(V.tabulate_dof_coordinates()))

    quad8 = NumpyQuad(
        np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]]
        )
    )
    shapes8 = quad8.interpolate(V.sub(0).collapse().tabulate_dof_coordinates(), (2,))
    assert np.isclose(np.sum(shapes8), len(V.tabulate_dof_coordinates()))

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
    x_dofs = V.sub(0).collapse().tabulate_dof_coordinates()
    x = x_dofs[:, 0]
    y = x_dofs[:, 1]

    def analytic(x, y):
        return x * y * (x - 1) * (y - 1) / 4

    xi = mapping(x, 0, 1)
    eta = mapping(y, 0, 1)
    shapes9 = quad9.interpolate(np.column_stack((xi, eta)), (1,))
    u = analytic(xi, eta)
    e = u - shapes9[0]
    print(np.allclose(shapes9[0], u))
    print(np.linalg.norm(e))
    assert np.linalg.norm(e) < 1e-14


if __name__ == "__main__":
    test()
