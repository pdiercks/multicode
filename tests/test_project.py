from mpi4py import MPI
from dolfinx import fem, mesh
from basix.ufl import element
from multi.projection import project, orthogonal_part, compute_proj_errors, compute_proj_errors_orth_basis
from multi.shapes import NumpyQuad
import numpy as np
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.algorithms.gram_schmidt import gram_schmidt


def test():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    ve = element("P", domain.basix_cell(), 2, shape=(2,))
    V = fem.functionspace(domain, ve)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    quad = NumpyQuad(nodes)
    shapes = quad.interpolate(V)
    Vdim = V.dofmap.bs * V.dofmap.index_map.size_global
    space = NumpyVectorSpace(Vdim)
    basis = space.from_numpy(shapes)

    alpha = np.arange(8)
    beta = np.array([0.0, 1.0, 2.0, 1.0, 4.0, 6.0, 7.0, 0.1])
    U = basis.lincomb(alpha)
    U.append(basis.lincomb(beta))
    U_orth = orthogonal_part(basis, U, product=None, orth=False)

    err = U_orth
    assert np.all(err.norm() < 1e-12)

    errors = compute_proj_errors(basis, U, None)
    assert np.isclose(errors[0], 1.)
    assert np.isclose(errors[-1], 0.)

    other = gram_schmidt(basis, product=None, copy=True)
    V_proj = project(other, U, product=None, orth=True)

    err_v = U - V_proj
    assert np.all(err_v.norm() < 1e-12)

    errors = compute_proj_errors_orth_basis(other, U, None)
    assert np.isclose(errors[0], 1.)
    assert np.isclose(errors[-1], 0.)


if __name__ == "__main__":
    test()
