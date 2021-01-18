"""test hierarchical shape functions"""
import dolfin as df
import numpy as np
from multi.shapes import get_hierarchical_shapes_2d


def test():
    mesh = df.UnitSquareMesh(8, 8)
    mesh.translate(df.Point((1.0, 1.0)))
    V = df.FunctionSpace(mesh, "CG", 2)
    N = get_hierarchical_shapes_2d(V, 2)
    points = [(1, 1), (2, 1), (2, 2), (1, 2)]
    edges = [(1.5, 1), (2, 1.5), (1.5, 2), (1, 1.5)]
    values = [1, 1, 1, 1]
    edge_values = [-1, -1, -1, -1]

    f = df.Function(V)
    g = df.Function(V)
    for k, (p, v) in enumerate(zip(points, values)):
        f.vector().set_local(N[k])
        assert np.isclose(f(p), v)
        other_points = set(points).difference([p])
        for op in other_points:
            assert np.isclose(f(op), 0)

    for k, (p, v) in enumerate(zip(edges, edge_values)):
        g.vector().set_local(N[k + 4])
        assert np.isclose(g(p), v)


if __name__ == "__main__":
    test()
