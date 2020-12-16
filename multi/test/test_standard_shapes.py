"""test standard shape functions"""
import dolfin as df
import numpy as np
from multi.shapes import NumpyLine, NumpyQuad4, NumpyQuad8


def test():
    mesh = df.UnitIntervalMesh(10)
    V = df.FunctionSpace(mesh, "CG", 2)

    line2 = NumpyLine(np.array([0, 1]), 1)
    line3 = NumpyLine(np.array([0, 1, 0.5]), 2)
    x_dofs = V.tabulate_dof_coordinates()
    n_verts = len(x_dofs)

    shapes = line2.interpolate(x_dofs, 1)
    assert np.isclose(np.sum(shapes), n_verts)
    shapes = line3.interpolate(x_dofs, 1)
    assert np.isclose(np.sum(shapes), n_verts)

    mesh = df.UnitSquareMesh(6, 6)
    V = df.VectorFunctionSpace(mesh, "CG", 1)

    gexpr = df.Expression(("(1 - x[0]) * (1 - x[1])", "0"), degree=1)
    g = df.interpolate(gexpr, V)

    quad = NumpyQuad4(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    shapes = quad.interpolate(V.sub(0).collapse().tabulate_dof_coordinates(), 2)

    assert np.allclose(shapes[0], g.vector()[:])
    assert np.isclose(np.sum(shapes), len(V.tabulate_dof_coordinates()))

    quad8 = NumpyQuad8(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]]))
    shapes = quad8.interpolate(V.sub(0).collapse().tabulate_dof_coordinates(), 2)

    assert np.isclose(np.sum(shapes), len(V.tabulate_dof_coordinates()))


if __name__ == "__main__":
    test()
