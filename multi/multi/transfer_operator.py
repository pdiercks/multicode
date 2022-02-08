# NOTE the implementation of the discretization of the transfer
# operator with Fenics is based on the code published along with [BS18]
# [BS18]: https://arxiv.org/abs/1706.09179

import time
import dolfin as df
import numpy as np
from scipy.sparse import linalg
import scipy.sparse as sp
from multi.misc import make_mapping
from pymor.operators.numpy import NumpyMatrixOperator


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
    full_operator = sp.csc_matrix(A.getValuesCSR()[::-1], shape=A.size)
    operator = full_operator[:, all_inner_dofs][all_inner_dofs, :]

    # factorization
    matrix_shape = operator.shape
    start = time.time()
    operator = linalg.factorized(operator)
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
        full_product_operator = sp.csc_matrix(P.getValuesCSR()[::-1], shape=P.size)

        source_product = NumpyMatrixOperator(
            full_product_operator[dirichlet_dofs, :][:, dirichlet_dofs]
        )
        range_product = NumpyMatrixOperator(full_product_operator[V_to_R, :][:, V_to_R])
    else:
        source_product = None
        range_product = None

    return (NumpyMatrixOperator(transfer_operator), source_product, range_product)
