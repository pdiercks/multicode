"""test serendipity shape functions"""

import dolfin as df
import numpy as np
from multi.shapes import NumpyQuad


def test():
    mesh = df.RectangleMesh(df.Point([0, 0]), df.Point([2, 2]), 20, 20, "crossed")
    mesh.translate(df.Point([-1, -1]))
    V = df.FunctionSpace(mesh, "CG", 2)
    quad8 = NumpyQuad(
        np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    )
    shapes = quad8.interpolate(V)

    def serendipity(xi, eta):
        """return shape function at 8th node"""
        return (1 - xi) * (1 - eta ** 2) / 2

    assert np.isclose(np.sum(shapes), len(V.tabulate_dof_coordinates()))
    x_dofs = V.tabulate_dof_coordinates()
    x = x_dofs[:, 0]
    y = x_dofs[:, 1]

    u = serendipity(x, y)
    e = u - shapes[7]
    assert np.linalg.norm(e) < 1e-14


if __name__ == "__main__":
    test()
