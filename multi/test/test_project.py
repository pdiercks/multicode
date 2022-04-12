from multi.projection import project
from multi.shapes import NumpyQuad
import dolfin as df
import numpy as np
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.algorithms.gram_schmidt import gram_schmidt


def test():
    mesh = df.UnitSquareMesh(10, 10)
    V = df.VectorFunctionSpace(mesh, "CG", 2)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    quad = NumpyQuad(nodes)
    shapes = quad.interpolate(V)
    space = NumpyVectorSpace(V.dim())
    basis = space.from_numpy(shapes)

    alpha = np.arange(8)
    beta = np.array([0.0, 1.0, 2.0, 1.0, 4.0, 6.0, 7.0, 0.1])
    U = basis.lincomb(alpha)
    U.append(basis.lincomb(beta))
    U_proj = project(basis, U, product=None, orth=False)

    err = U - U_proj
    assert np.all(err.norm() < 1e-12)

    other = gram_schmidt(basis, product=None, copy=True)
    V_proj = project(other, U, product=None, orth=True)

    err_v = U - V_proj
    assert np.all(err_v.norm() < 1e-12)


if __name__ == "__main__":
    test()
