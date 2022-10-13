import numpy as np
from multi.misc import set_values
from pymor.bindings.fenics import FenicsVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


# def test_fenics():
#     mesh = df.UnitSquareMesh(6, 6)
#     V = df.FunctionSpace(mesh, "CG", 1)
#     space = FenicsVectorSpace(V)
#     num_vecs = 10
#     U = space.random(num_vecs, distribution="normal")
#     bc = df.DirichletBC(V, df.Constant(0.0), df.DomainBoundary())
#     bc_dofs = np.array(list(bc.get_boundary_values().keys()))
#     all_dofs = np.arange(V.dim())
#     inner_dofs = np.setdiff1d(all_dofs, bc_dofs)
#     values = np.ones((num_vecs, bc_dofs.size))
#     R = set_values(U, bc_dofs, values)
#     assert np.sum(R.dofs(bc_dofs)) == len(bc_dofs) * num_vecs
#     assert np.allclose(R.dofs(inner_dofs), U.dofs(inner_dofs))


def test_numpy():
    # numpy
    space = NumpyVectorSpace(100)
    num_vecs = 10
    U = space.random(num_vecs, distribution="normal")
    dofs = np.array([0, 5, 99, 53, 44, 12, 17, 64])
    values = np.ones((num_vecs, dofs.size))
    set_values(U, dofs, values)
    assert np.sum(U.dofs(dofs)) == len(dofs) * num_vecs


if __name__ == "__main__":
    test_numpy()
    # test_fenics()
