import numpy as np
import dolfin as df

from multi.extension import extend_pymor
from multi.misc import locate_dofs
from multi.product import InnerProduct
from multi.shapes import NumpyQuad, get_hierarchical_shape_functions

from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.special import erfinv

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenics import FenicsVectorSpace
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray


def construct_hierarchical_basis(
    problem,
    max_degree,
    solver_options=None,
    orthonormalize=False,
    product=None,
    return_edge_basis=False,
):
    """construct hierarchical basis (full space)

    Parameters
    ----------
    problem : multi.problems.LinearProblemBase
        A suitable problem for which to compute hierarchical
        edge basis functions.
    max_degree : int
        The maximum polynomial degree of the shape functions.
        Must be greater than or equal to 2.
    solver_options : dict, optional
        Solver options in pymor format.
    orthonormalize : bool, optional
        If True, orthonormalize the edge basis to inner ``product``.
    product : optional
        Inner product wrt to which the edge basis is orthonormalized
        if ``orthonormalize`` is True.

    Returns
    -------
    basis : VectorArray
        The hierarchical basis extended into the interior of
        the domain of the problem.
    edge_basis : VectorArray
        The hierarchical edge basis (if ``return_edge_basis`` is True).

    """
    V = problem.V
    try:
        edge_spaces = problem.edge_spaces
    except AttributeError as err:
        raise err("There are no edge spaces defined for given problem.")

    # ### construct the edge basis on the bottom edge
    ufl_element = V.ufl_element()
    L = edge_spaces[0]
    x_dofs = L.sub(0).collapse().tabulate_dof_coordinates()
    edge_basis = get_hierarchical_shape_functions(
        x_dofs[:, 0], max_degree, ncomp=ufl_element.value_size()
    )
    source = FenicsVectorSpace(L)
    B = source.from_numpy(edge_basis)

    # ### build inner product for edge space
    product_bc = df.DirichletBC(L, df.Function(L), df.DomainBoundary())
    inner_product = InnerProduct(L, product, bcs=(product_bc,))
    product = inner_product.assemble_operator()

    if orthonormalize:
        gram_schmidt(B, product=product, copy=False)

    # ### initialize boundary data
    basis_length = len(B)
    Vdim = V.dim()
    boundary_data = np.zeros((basis_length * len(edge_spaces), Vdim))

    def mask(index):
        start = index * basis_length
        end = (index + 1) * basis_length
        return np.s_[start:end]

    # ### fill in values for boundary data
    for i in range(len(edge_spaces)):
        boundary_data[mask(i), problem.V_to_L[i]] = B.to_numpy()

    # ### extend edge basis into the interior of the domain
    basis = extend_pymor(problem, boundary_data, solver_options=solver_options)
    if return_edge_basis:
        return basis, B
    else:
        return basis


def compute_phi(problem, solver_options=None):
    # just a wrapper around extend
    # only need to define the boundary data
    V = problem.V
    n_vertices = 4  # assumes rectangular domain
    nodes = problem.domain.get_nodes(n=n_vertices)
    quadrilateral = NumpyQuad(nodes)
    shape_functions = quadrilateral.interpolate(V)

    bc = df.DirichletBC(V, df.Function(V), df.DomainBoundary())
    bc_dofs = list(bc.get_boundary_values().keys())
    boundary_data = np.zeros_like(shape_functions)
    boundary_data[:, bc_dofs] = shape_functions[:, bc_dofs]

    phi = extend_pymor(problem, boundary_data, solver_options=solver_options)
    return phi


# adapted from pymor.algorithms.randrangefinder.adaptive_rrf
# to be used with multi.oversampling.OversamplingProblem
def _compute_fine_scale_snapshots_os(
    oversampling_problem,
    coarse_scale_basis,
    source_product=None,
    range_product=None,
    tol=1e-4,
    failure_tolerance=1e-15,
    num_testvecs=20,
    lambda_min=None,
    train_vectors=None,
):
    r"""Adaptive randomized range approximation of `A`.
    This is an implementation of Algorithm 1 in [BS18]_.

    The following modifications where made:
        - instead of a transfer operator A, the full oversampling
        problem is used
        - optional training data was added (`train_vectors`)

    The image Av = A.apply(v) is equivalent to the restriction
    of the full solution to the target domain Ω_in, i.e.
        U = oversampling_problem.solve(v)
    See multi.oversampling.OversamplingProblem.solve.

    Given the |Operator| `A`, the return value of this method is the |VectorArray|
    `B` with the property

    .. math::
        \Vert A - P_{span(B)} A \Vert \leq tol

    with a failure probability smaller than `failure_tolerance`, where the norm denotes the
    operator norm. The inner product of the range of `A` is given by `range_product` and
    the inner product of the source of `A` is given by `source_product`.

    Parameters
    ----------
    oversampling_problem
        The full oversampling problem associated with a (transfer) |Operator| A.
    coarse_scale_basis
        The |VectorArray| containing the coarse scale basis.
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

    Returns
    -------
    B
        |VectorArray| which contains fine scale part of the computed solutions
        of the oversampling problem.
    """
    problem = oversampling_problem

    assert source_product is None or isinstance(source_product, Operator)
    assert range_product is None or isinstance(range_product, Operator)
    assert (
        train_vectors is None
        or isinstance(train_vectors, VectorArray)
        and train_vectors.space is problem.source
    )
    assert coarse_scale_basis.space is problem.range

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

    # NOTE problem.source is the full space, while the source product is of lower dimension
    num_source_dofs = len(problem._bc_dofs_gamma_out)
    testfail = failure_tolerance / min(num_source_dofs, problem.range.dim)
    testlimit = (
        np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * tol
    )
    maxnorm = np.inf
    # set of test vectors
    R = problem.generate_random_boundary_data(num_testvecs, distribution="normal")
    M = problem.solve(R)

    # get vertex dofs of target subdomain
    subdomain = problem.subdomain_problem.domain
    vertices = subdomain.get_nodes(n=4)  # assumes rectangular domain
    vertex_dofs = locate_dofs(problem.range.V.tabulate_dof_coordinates(), vertices)

    B = problem.range.empty()
    snapshots = problem.range.empty()
    while maxnorm > testlimit:
        basis_length = len(B)
        if train_vectors is not None and basis_length < len(train_vectors):
            v = train_vectors[basis_length]
        else:
            v = problem.generate_random_boundary_data(count=1, distribution="normal")

        # subtract coarse scale part
        r = problem.solve(v)
        nodal_values = r.dofs(vertex_dofs)
        r -= coarse_scale_basis.lincomb(nodal_values)
        snapshots.append(r)

        B.append(problem.solve(v))
        gram_schmidt(
            B,
            range_product,
            atol=0,
            rtol=0,
            offset=basis_length,
            copy=False,
        )
        M -= B.lincomb(B.inner(M, range_product).T)
        maxnorm = np.max(M.norm(range_product))

    zero_at_dofs = np.allclose(
        snapshots.to_numpy()[:, vertex_dofs], np.zeros(vertex_dofs.size)
    )
    assert zero_at_dofs

    return snapshots


def compute_fine_scale_snapshots(
    A,
    coarse_scale_basis,
    range_space,
    source_product=None,
    range_product=None,
    tol=1e-4,
    failure_tolerance=1e-15,
    num_testvecs=20,
    lambda_min=None,
    train_vectors=None,
    iscomplex=False,
    check_ortho=True,
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
        The (transfer) |Operator| A.
    coarse_scale_basis
        The |VectorArray| containing the coarse scale basis.
    range_space
        The dolfin.FunctionSpace representing the range space.
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
    check_ortho
        If `True`, in modified Gram-Schmidt algorithm check if
        the resulting |VectorArray| is really orthonormal.

    Returns
    -------
    B
        |VectorArray| which contains fine scale part of the training vectors.
    """

    assert source_product is None or isinstance(source_product, Operator)
    assert range_product is None or isinstance(range_product, Operator)
    assert (
        train_vectors is None
        or isinstance(train_vectors, VectorArray)
        and train_vectors.space is A.source
    )
    assert isinstance(A, Operator)
    assert coarse_scale_basis.space is A.range

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

    # assumes target subdomain is rectangle
    mesh = range_space.mesh()
    coord = mesh.coordinates()
    xmin = np.amin(coord[:, 0])
    xmax = np.amax(coord[:, 0])
    ymin = np.amin(coord[:, 1])
    ymax = np.amax(coord[:, 1])
    nodes = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    vertex_dofs = locate_dofs(range_space.tabulate_dof_coordinates(), nodes)

    B = A.range.empty()
    snapshots = A.range.empty()
    while maxnorm > testlimit:
        basis_length = len(B)
        if train_vectors is not None and basis_length < len(train_vectors):
            v = train_vectors[basis_length]
        else:
            v = A.source.random(distribution="normal")
            if iscomplex:
                v += 1j * A.source.random(distribution="normal")

        # subtract coarse scale part
        r = A.apply(v)
        nodal_values = r.dofs(vertex_dofs)
        r -= coarse_scale_basis.lincomb(nodal_values)
        snapshots.append(r)

        B.append(A.apply(v))
        gram_schmidt(
            B,
            range_product,
            atol=0,
            rtol=0,
            offset=basis_length,
            check=check_ortho,
            copy=False,
        )
        M -= B.lincomb(B.inner(M, range_product).T)
        maxnorm = np.max(M.norm(range_product))

    zero_at_dofs = np.allclose(
        snapshots.to_numpy()[:, vertex_dofs], np.zeros(vertex_dofs.size)
    )
    assert zero_at_dofs

    return snapshots
