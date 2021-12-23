import time
import dolfin as df
import numpy as np
from scipy.sparse import linalg
import scipy.sparse as sp
from multi import Domain, LinearElasticityProblem
from multi.misc import make_mapping


def average_remover(size):
    return np.eye(size) - np.ones((size, size)) / float(size)


# NOTE this implementation is based on the code published along with [BS18]
# [BS18]: https://arxiv.org/abs/1706.09179
def transfer_operator_subdomains_2d(problem, subdomain, ar=False):
    """Discretize the transfer operator

    Parameters
    ----------
    problem : multi.LinearElasticityProblem
        A suitable oversampling problem.
    subdomain : multi.Domain
        The target subdomain.

    Returns
    -------
    tranfer_operator :

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
    return transfer_operator
