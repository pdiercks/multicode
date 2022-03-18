import numpy as np
import dolfin as df

from multi.extension import extend_pymor
from multi.shapes import NumpyQuad
from multi.misc import locate_dofs

from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.special import erfinv

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenics import (
    FenicsMatrixOperator,
    FenicsVisualizer,
)
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.operators.interface import Operator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.vectorarrays.interface import VectorArray


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


def construct_coarse_scale_basis(problem, solver_options=None, return_fom=False):
    """construct coarse scale basis for given subdomain problem

    Parameters
    ----------
    problem : a suitable problem like :class:`~multi.linear_elasticity.LinearElasticityProblem`
    solver_options : dict, optional
        A dict of strings. See https://github.com/pymor/pymor/blob/2021.2.0/src/pymor/operators/interface.py
    return_fom : bool, optional
        If True, the full order model is returned.

    Returns
    -------
    basis : Fenics VectorArray
        The coarse scale basis for given subdomain.
    fom
        The full order model (if `return_fom` is `True`).

    """
    # full system matrix
    A = df.assemble(problem.get_lhs())

    # ### define bilinear shape functions as inhomogeneous dirichlet bcs
    V = problem.V
    f = df.Function(V)  # placeholder for rhs vector operators
    g = df.Function(V)  # boundary data
    n_vertices = 4  # assumes rectangular domain
    nodes = problem.domain.get_nodes(n=n_vertices)
    quadrilateral = NumpyQuad(nodes)
    shape_functions = quadrilateral.interpolate(V)

    null = np.zeros(V.dim())
    # source = problem.source
    vector_operators = []
    for shape in shape_functions:
        f.vector().set_local(null)
        g.vector().set_local(shape)
        bc = df.DirichletBC(V, g, df.DomainBoundary())
        A_bc = A.copy()
        bc.zero_columns(A_bc, f.vector(), 1.0)
        vector_operators.append(
            VectorOperator(problem.range.make_array([f.vector().copy()]))
        )

    lhs = FenicsMatrixOperator(A_bc, V, V, solver_options=solver_options)
    parameter_functionals = [
        ProjectionParameterFunctional("mu", shape_functions.shape[0], index=i)
        for i in range(shape_functions.shape[0])
    ]
    rhs = LincombOperator(vector_operators, parameter_functionals)

    # ### inner products
    energy_mat = A.copy()
    energy_0_mat = A_bc.copy()
    l2_mat = problem.get_product(name="l2", bcs=False)
    l2_0_mat = l2_mat.copy()
    h1_mat = problem.get_product(name="h1", bcs=False)
    h1_0_mat = h1_mat.copy()
    bc.apply(l2_0_mat)
    bc.apply(h1_0_mat)

    fom = StationaryModel(
        lhs,
        rhs,
        output_functional=None,
        products={
            "energy": FenicsMatrixOperator(energy_mat, V, V, name="energy"),
            "energy_0": FenicsMatrixOperator(energy_0_mat, V, V, name="energy_0"),
            "l2": FenicsMatrixOperator(l2_mat, V, V, name="l2"),
            "l2_0": FenicsMatrixOperator(l2_0_mat, V, V, name="l2_0"),
            "h1": FenicsMatrixOperator(h1_mat, V, V, name="h1"),
            "h1_0": FenicsMatrixOperator(h1_0_mat, V, V, name="h1_0"),
        },
        error_estimator=None,
        visualizer=FenicsVisualizer(problem.range),
        name="FOM",
    )

    # ### compute the coarse scale basis
    dim = 2  # spatial dimension
    z = dim * n_vertices
    basis = fom.operator.source.empty(reserve=z)
    Identity = np.eye(z)
    for row in Identity:
        basis.append(fom.solve({"mu": row}))

    if return_fom:
        return basis, fom
    else:
        return basis


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
