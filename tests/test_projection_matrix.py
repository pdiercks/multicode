from mpi4py import MPI
from dolfinx import mesh, fem
from basix.ufl import element
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from multi.product import InnerProduct
from multi.projection import orthogonal_part
from multi.solver import build_nullspace
import numpy as np
import pytest


@pytest.mark.parametrize("product_name",["euclidean", "h1"])
def test(product_name):
    nx = ny = 10
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
    fe_deg = 1
    fe = element("P", domain.basix_cell(), fe_deg, shape=(2,))
    V = fem.functionspace(domain, fe)

    source = FenicsxVectorSpace(V)
    product = None
    W = np.eye(source.dim)
    if not product_name == "euclidean":
        product = InnerProduct(V, product="h1")
        product_mat = product.assemble_matrix()
        product = FenicsxMatrixOperator(product_mat, V, V)
        W = product.matrix[:, :]
    basis = build_nullspace(source, product=product)

    # projection onto nullspace
    # A holds rigid body modes as columns
    # W is weighting matrix due to inner product
    # right = np.dot(A.T, W) # inner product
    # middle = np.linalg.inv(np.dot(A.T, np.dot(W, A))) # gramian
    # left = A # form linear combination
    # P = np.dot(left, np.dot(middle, right))
    # by design the gramian should be the identity
    # (user is to blame if working with non-orthonormal basis?)

    A = basis.to_numpy().T # pymor stores row vectors
    P = np.dot(A, np.dot(A.T, W))

    # test data in span of basis
    dummy_null_space = NumpyVectorSpace(3)
    coeff = dummy_null_space.random(1)
    U = basis.lincomb(coeff.to_numpy()).to_numpy().flatten()
    U_proj = P.dot(U)
    assert np.allclose(U, U_proj)

    # test data not in span of basis
    # compare to alternative method of computing orthogonal part
    U = source.random(1)
    r1 = orthogonal_part(basis, U, product, orth=True).to_numpy().flatten()
    r2 = (np.eye(P.shape[0]) - P).dot(U.to_numpy().flatten())
    assert np.allclose(r1, r2)



if __name__ == "__main__":
    test("h1")
