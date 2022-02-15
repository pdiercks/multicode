# NOTE the implementation of the discretization of the transfer
# operator with Fenics is based on the code published along with [BS18]
# [BS18]: https://arxiv.org/abs/1706.09179

import time
import dolfin as df
import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh, factorized, LinearOperator
from scipy.special import erfinv

from multi.misc import make_mapping

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorArray


# modified version of `pymor.algorithms.randrangefinder.adaptive_rrf`
# source: https://github.com/pymor/pymor/blob/main/src/pymor/algorithms/randrangefinder.py#L15-#L93
def modified_adaptive_rrf(
    A,
    source_product=None,
    range_product=None,
    tol=1e-4,
    failure_tolerance=1e-15,
    num_testvecs=20,
    lambda_min=None,
    train_vectors=None,
    iscomplex=False,
):
    r"""Adaptive randomized range approximation of `A`.

    This is an implementation of Algorithm 1 in [BS18]_.
    It was modified to use a predefined set of source vectors `train_vectors` as
    training data for the first `len(train_vectors)` basis functions.

    Given the |Operator| `A`, the return value of this method is the |VectorArray|
    `B` with the property

    .. math::
        \Vert A - P_{span(B)} A \Vert \leq tol

    with a failure probability smaller than `failure_tolerance`, where the norm denotes the
    operator norm. The inner product of the range of `A` is given by `range_product` and
    the inner product of the source of `A` is given by `source_product`.

    Parameters
    ----------
    A
        The |Operator| A.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        Inner product |Operator| of the range of A.
    tol
        Error tolerance for the algorithm.
    failure_tolerance
        Maximum failure probability.
    num_testvecs
        Number of test vectors.
    lambda_min
        The smallest eigenvalue of source_product.
        If `None`, the smallest eigenvalue is computed using scipy.
    train_vectors
        |VectorArray| containing a set of predefined training
        vectors.
    iscomplex
        If `True`, the random vectors are chosen complex.

    Returns
    -------
    B
        |VectorArray| which contains the basis, whose span approximates the range of A.
    """

    assert source_product is None or isinstance(source_product, Operator)
    assert range_product is None or isinstance(range_product, Operator)
    assert (
        train_vectors is None
        or isinstance(train_vectors, VectorArray)
        and train_vectors.space is A.source
    )
    assert isinstance(A, Operator)

    B = A.range.empty()

    R = A.source.random(num_testvecs, distribution="normal")
    if iscomplex:
        R += 1j * A.source.random(num_testvecs, distribution="normal")

    if source_product is None:
        lambda_min = 1
    elif lambda_min is None:

        def mv(v):
            return source_product.apply(source_product.source.from_numpy(v)).to_numpy()

        def mvinv(v):
            return source_product.apply_inverse(
                source_product.range.from_numpy(v)
            ).to_numpy()

        L = LinearOperator(
            (source_product.source.dim, source_product.range.dim), matvec=mv
        )
        Linv = LinearOperator(
            (source_product.range.dim, source_product.source.dim), matvec=mvinv
        )
        lambda_min = eigsh(
            L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv
        )[0]

    testfail = failure_tolerance / min(A.source.dim, A.range.dim)
    testlimit = (
        np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * tol
    )
    maxnorm = np.inf
    M = A.apply(R)

    while maxnorm > testlimit:
        basis_length = len(B)
        if train_vectors is not None and basis_length < len(train_vectors):
            v = train_vectors[basis_length]
        else:
            v = A.source.random(distribution="normal")
            if iscomplex:
                v += 1j * A.source.random(distribution="normal")
        B.append(A.apply(v))
        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)
        M -= B.lincomb(B.inner(M, range_product).T)
        maxnorm = np.max(M.norm(range_product))

    return B


def average_remover(size):
    return np.eye(size) - np.ones((size, size)) / float(size)


def transfer_operator_subdomains_2d(problem, subdomain, product=None, ar=False):
    """Discretize the transfer operator

    Parameters
    ----------
    problem : multi.LinearElasticityProblem
        A suitable oversampling problem.
    subdomain : multi.Domain
        The target subdomain.
    product : optional, str
        The inner product of range and source space.
        See multi.product.InnerProduct for possible values.

    Returns
    -------
    tranfer_operator : NumpyMatrixOperator
        The transfer operator matrix for the given problem and target subdomain.
    source_product : NumpyMatrixOperator or None
        Inner product matrix of the source space.
    range_product : NumpyMatrixOperator or None
        Inner product matrix of the range space.

    """
    # full space
    V = problem.V
    # range space
    R = df.FunctionSpace(subdomain.mesh, V.ufl_element())
    V_to_R = make_mapping(R, V)

    # dofs
    all_dofs = np.arange(V.dim())
    bcs = df.DirichletBC(V, df.Constant((0.0, 0.0)), df.DomainBoundary())
    dirichlet_dofs = np.array(list(bcs.get_boundary_values().keys()))
    all_inner_dofs = np.setdiff1d(all_dofs, dirichlet_dofs)

    A = df.as_backend_type(df.assemble(problem.get_lhs())).mat()
    full_operator = csc_matrix(A.getValuesCSR()[::-1], shape=A.size)
    operator = full_operator[:, all_inner_dofs][all_inner_dofs, :]

    # factorization
    matrix_shape = operator.shape
    start = time.time()
    operator = factorized(operator)
    end = time.time()
    print(f"factorization of {matrix_shape} matrix in {end-start}")

    # mapping from old to new dof numbers
    newdofs = np.zeros((V.dim(),), dtype=int)
    newdofs[all_inner_dofs] = np.arange(all_inner_dofs.size)
    range_dofs = newdofs[V_to_R]

    rhs_op = full_operator[:, dirichlet_dofs][all_inner_dofs, :]
    start = time.time()
    transfer_operator = -operator(rhs_op.todense())[range_dofs, :]
    end = time.time()
    print(f"applied operator to rhs in {end-start}")

    if ar:
        ar = average_remover(transfer_operator.shape[0])
        transfer_operator = ar.dot(transfer_operator)

    # compute inner products
    if product is not None:
        inner_product = problem.get_product(name=product, bcs=False)
        P = df.as_backend_type(inner_product).mat()
        full_product_operator = csc_matrix(P.getValuesCSR()[::-1], shape=P.size)

        source_product = NumpyMatrixOperator(
            full_product_operator[dirichlet_dofs, :][:, dirichlet_dofs]
        )
        range_product = NumpyMatrixOperator(full_product_operator[V_to_R, :][:, V_to_R])
    else:
        source_product = None
        range_product = None

    return (NumpyMatrixOperator(transfer_operator), source_product, range_product)
