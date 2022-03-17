import numpy as np
import dolfin as df
from multi.misc import set_values
from pymor.bindings.fenics import FenicsVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


def test():
    # numpy
    space = NumpyVectorSpace(100)
    num_vecs = 10
    U = space.random(num_vecs, distribution="normal")
    dofs = np.array([0, 5, 99, 53, 44, 12, 17, 64])
    values = np.ones((num_vecs, dofs.size))
    set_values(U, dofs, values)
    assert np.sum(U.dofs(dofs)) == len(dofs) * num_vecs


if __name__ == "__main__":
    test()
