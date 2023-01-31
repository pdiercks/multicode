import dolfinx
import numpy as np
from petsc4py.PETSc import ScalarType

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.tools.timing import Timer
from pymor.reductors.basic import extend_basis

from scipy.linalg import eigh
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.special import erfinv

from multi.product import InnerProduct
from multi.dofmap import QuadrilateralDofLayout
from multi.sampling import correlation_function
from multi.shapes import NumpyLine


# modified version of pymor.algorithms.rand_la.adaptive_rrf
def adaptive_edge_rrf_normal(
    transfer_problem,
    random_state,
    active_edges,
    source_product=None,
    range_product=None,
    error_tol=1e-4,
    failure_tolerance=1e-15,
    num_testvecs=20,
    lambda_min=None,
    **sampling_options,
):
    r"""Adaptive randomized range approximation of `A`.
    This is an implementation of Algorithm 1 in [BS18]_.

    Given the |Operator| `A`, the return value of this method is the
    |VectorArray| `B` with the property

    .. math::
        \Vert A - P_{span(B)} A \Vert \leq tol

    with a failure probability smaller than `failure_tolerance`, where the
    norm denotes the operator norm. The inner product of the range of
    `A` is given by `range_product` and
    the inner product of the source of `A` is given by `source_product`.

    NOTE
    ----
    Instead of a transfer operator A, a transfer problem is used.
    (see multi.problem.TransferProblem)
    The image Av = A.apply(v) is equivalent to the restriction
    of the full solution to the target domain Ω_in, i.e.
        U = transfer_problem.solve(v)


    Parameters
    ----------
    transfer_problem
        The transfer problem associated with a (transfer) |Operator| A.
    random_state
        The random state to generate samples.
    active_edges
        A list of edges of the target subdomain.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        A str specifying which inner product to use.
    error_tol
        Error tolerance for the algorithm.
    failure_tolerance
        Maximum failure probability.
    num_testvecs
        Number of test vectors.
    lambda_min
        The smallest eigenvalue of source_product.
        If `None`, the smallest eigenvalue is computed using scipy.
    sampling_options
        Optional keyword arguments for the generation of
        random samples (training data).
        see `_create_random_values`.

    Returns
    -------
    pod_bases
        A dict which contains a |VectorArray| for each 'active edge'.
        The |VectorArray| contains the POD basis which
        span approximates the image of the transfer operator A
        restricted to the respective edge.
    range_products
        The inner product operators constructed in the edge
        range spaces.

    """

    logger = getLogger("multi.range_finder.adaptive_edge_rrf", level="DEBUG")
    tp = transfer_problem

    timer = Timer("rrf")
    distribution = "normal"

    assert source_product is None or isinstance(source_product, Operator)
    # TODO remove arg range_product or let this be a list
    # assert range_product is None or isinstance(range_product, Operator)

    if source_product is None:
        lambda_min = 1
    elif lambda_min is None:

        def mv(v):
            return source_product.apply(
                    source_product.source.from_numpy(v)).to_numpy()

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

    # ### test set
    R = tp.generate_random_boundary_data(
            count=num_testvecs, distribution=distribution,
            random_state=random_state
            )
    M = tp.solve(R)

    dof_layout = QuadrilateralDofLayout()
    edge_index_map = dof_layout.local_edge_index_map

    # ### initialize data structures
    test_set = {}
    range_spaces = {}
    range_products = {}
    pod_bases = {}
    maxnorm = np.array([], dtype=float)
    edges = np.array([], dtype=str)
    coarse_basis = {}
    # the dofs for vertices on the boundary of the edge
    edge_boundary_dofs = {}

    timer.start()
    for i in range(dof_layout.num_entities[1]):
        edge = edge_index_map[i]
        edges = np.append(edges, edge)

        edge_mesh = tp.subproblem.domain.fine_edge_grid[edge]
        edge_space = tp.subproblem.edge_spaces["fine"][edge]
        range_spaces[edge] = FenicsxVectorSpace(edge_space)

        # ### create dirichletbc for range product
        facet_dim = edge_mesh.topology.dim - 1
        vertices = dolfinx.mesh.locate_entities_boundary(
                edge_mesh, facet_dim, lambda x: np.full(
                    x[0].shape, True, dtype=bool)
                )
        _dofs_ = dolfinx.fem.locate_dofs_topological(
                edge_space, facet_dim, vertices)
        gdim = tp.subproblem.domain.grid.geometry.dim
        range_bc = dolfinx.fem.dirichletbc(
                np.array((0, ) * gdim, dtype=ScalarType), _dofs_, edge_space)
        edge_boundary_dofs[edge] = range_bc.dof_indices()[0]

        # ### range product
        inner_product = InnerProduct(
                edge_space, range_product, bcs=(range_bc, ))
        range_product_op = FenicsxMatrixOperator(
                inner_product.assemble_matrix(), edge_space, edge_space)
        range_products[edge] = range_product_op

        # ### compute coarse scale edge basis
        nodes = dolfinx.mesh.compute_midpoints(edge_mesh, facet_dim, vertices)
        nodes = np.around(nodes, decimals=3)

        if edge in ("bottom", "top"):
            component = 0
        elif edge in ("left", "right"):
            component = 1

        line_element = NumpyLine(nodes[:, component])
        shape_funcs = line_element.interpolate(edge_space, component)
        N = range_spaces[edge].from_numpy(shape_funcs)
        coarse_basis[edge] = N

        # ### edge test sets
        dofs = tp.subproblem.V_to_L[edge]
        test_set[edge] = range_spaces[edge].from_numpy(M.dofs(dofs))
        # subtract coarse scale part
        test_cvals = test_set[edge].dofs(edge_boundary_dofs[edge])
        test_set[edge] -= N.lincomb(test_cvals)

        # ### initialize maxnorm
        if edge in active_edges:
            maxnorm = np.append(maxnorm, np.inf)
            # ### pod bases
            pod_bases[edge] = range_spaces[edge].empty()
        else:
            maxnorm = np.append(maxnorm, 0.0)

    timer.stop()
    logger.debug(f"Preparing stuff took t={timer.dt}s.")

    # NOTE tp.source is the full space, while the source product
    # is of lower dimension
    num_source_dofs = len(tp._bc_dofs_gamma_out)
    testfail = np.array(
            [failure_tolerance / min(num_source_dofs, space.dim)
             for space in range_spaces.values()])
    testlimit = (
        np.sqrt(2.0 * lambda_min) * erfinv(
            testfail ** (1.0 / num_testvecs)) * error_tol
    )

    logger.info(f"{lambda_min=}")
    logger.info(f"{testlimit=}")

    num_solves = 0
    while np.any(maxnorm > testlimit):

        v = tp.generate_random_boundary_data(
                1, distribution, random_state, **sampling_options)

        U = tp.solve(v)
        num_solves += 1

        target_edges = edges[maxnorm > testlimit]
        for edge in target_edges:
            B = pod_bases[edge]
            edge_space = range_spaces[edge]
            # restrict the training sample to the edge
            Udofs = edge_space.from_numpy(U.dofs(tp.subproblem.V_to_L[edge]))
            coarse_values = Udofs.dofs(edge_boundary_dofs[edge])
            U_fine = Udofs - coarse_basis[edge].lincomb(coarse_values)

            # extend pod basis
            extend_basis(U_fine, B, product=range_products[edge],
                         method="pod", pod_modes=1)

            # orthonormalize test set wrt pod basis
            M = test_set[edge]
            M -= B.lincomb(B.inner(M, range_products[edge]).T)

            norm = M.norm(range_products[edge])
            maxnorm[edge_index_map[edge]] = np.max(norm)

        logger.info(f"{maxnorm=}")

    return pod_bases, range_products, num_solves


def get_sigma_neig(dist, lc, D, rtol=5e-2):
    """return covariance matrix Σ and number of samples to be drawn
    using Σ"""
    Σ_exp = correlation_function(
            dist, lc, function_type="exponential"
            )
    Σ = D.dot(csc_matrix(Σ_exp).dot(D))
    # get largest eigenvalue
    lambda_max = eigsh(Σ, k=1, which="LM", return_eigenvectors=False)
    # determine subset of eigenvalues by value
    eigvals = eigh(Σ.toarray(), eigvals_only=True, turbo=True,
                   subset_by_value=[lambda_max * rtol, np.inf])
    num_eigv = eigvals.size
    return Σ, num_eigv


# modified version of pymor.algorithms.rand_la.adaptive_rrf
def adaptive_edge_rrf_mvn(
    transfer_problem,
    random_state,
    active_edges,
    source_product=None,
    range_product=None,
    error_tol=1e-4,
    failure_tolerance=1e-15,
    num_testvecs=20,
    lambda_min=None,
    **sampling_options,
):
    r"""Adaptive randomized range approximation of `A`.
    This is an implementation of Algorithm 1 in [BS18]_.

    Given the |Operator| `A`, the return value of this method is
    the |VectorArray| `B` with the property

    .. math::
        \Vert A - P_{span(B)} A \Vert \leq tol

    with a failure probability smaller than `failure_tolerance`, where the
    norm denotes the operator norm. The inner product of the range of `A`
    is given by `range_product` and
    the inner product of the source of `A` is given by `source_product`.

    NOTE
    ----
    Instead of a transfer operator A, a transfer problem is used.
    (see multi.problem.TransferProblem)
    The image Av = A.apply(v) is equivalent to the restriction
    of the full solution to the target domain Ω_in, i.e.
        U = transfer_problem.solve(v)


    Parameters
    ----------
    transfer_problem
        The transfer problem associated with a (transfer) |Operator| A.
    random_state
        The random state to generate samples.
    distribution
        The distribution to generate samples from.
    active_edges
        A list of edges of the target subdomain.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        A str specifying which inner product to use.
    error_tol
        Error tolerance for the algorithm.
    failure_tolerance
        Maximum failure probability.
    num_testvecs
        Number of test vectors.
    lambda_min
        The smallest eigenvalue of source_product.
        If `None`, the smallest eigenvalue is computed using scipy.
    sampling_options
        Optional keyword arguments for the generation of
        random samples (training data).
        see `_create_random_values`.

    Returns
    -------
    pod_bases
        A dict which contains a |VectorArray| for each 'active edge'.
        The |VectorArray| contains the POD basis which
        span approximates the image of the transfer operator A
        restricted to the respective edge.
    range_products
        The inner product operators constructed in the edge
        range spaces.

    """

    logger = getLogger("multi.range_finder.adaptive_edge_rrf", level="DEBUG")
    tp = transfer_problem

    timer = Timer("rrf")
    distribution = "multivariate_normal"

    assert source_product is None or isinstance(source_product, Operator)
    # TODO remove arg range_product or let this be a list
    # assert range_product is None or isinstance(range_product, Operator)

    if source_product is None:
        lambda_min = 1
    elif lambda_min is None:

        def mv(v):
            return source_product.apply(
                    source_product.source.from_numpy(v)).to_numpy()

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

    # ### test set
    R = tp.generate_random_boundary_data(
            count=num_testvecs, distribution="normal",
            random_state=random_state
            )
    M = tp.solve(R)

    dof_layout = QuadrilateralDofLayout()
    edge_index_map = dof_layout.local_edge_index_map

    # ### initialize data structures
    test_set = {}
    range_spaces = {}
    range_products = {}
    pod_bases = {}
    maxnorm = np.array([], dtype=float)
    edges = np.array([], dtype=str)
    coarse_basis = {}
    # the dofs for vertices on the boundary of the edge
    edge_boundary_dofs = {}

    timer.start()
    for i in range(dof_layout.num_entities[1]):
        edge = edge_index_map[i]
        edges = np.append(edges, edge)

        edge_mesh = tp.subproblem.domain.fine_edge_grid[edge]
        edge_space = tp.subproblem.edge_spaces["fine"][edge]
        range_spaces[edge] = FenicsxVectorSpace(edge_space)

        # ### create dirichletbc for range product
        facet_dim = edge_mesh.topology.dim - 1
        vertices = dolfinx.mesh.locate_entities_boundary(
                edge_mesh, facet_dim, lambda x: np.full(
                    x[0].shape, True, dtype=bool)
                )
        _dofs_ = dolfinx.fem.locate_dofs_topological(
                edge_space, facet_dim, vertices)
        gdim = tp.subproblem.domain.grid.geometry.dim
        range_bc = dolfinx.fem.dirichletbc(
                np.array((0, ) * gdim, dtype=ScalarType), _dofs_, edge_space)
        edge_boundary_dofs[edge] = range_bc.dof_indices()[0]

        # ### range product
        inner_product = InnerProduct(
                edge_space, range_product, bcs=(range_bc, ))
        range_product_op = FenicsxMatrixOperator(
                inner_product.assemble_matrix(), edge_space, edge_space)
        range_products[edge] = range_product_op

        # ### compute coarse scale edge basis
        nodes = dolfinx.mesh.compute_midpoints(edge_mesh, facet_dim, vertices)
        nodes = np.around(nodes, decimals=3)

        if edge in ("bottom", "top"):
            component = 0
        elif edge in ("left", "right"):
            component = 1

        line_element = NumpyLine(nodes[:, component])
        shape_funcs = line_element.interpolate(edge_space, component)
        N = range_spaces[edge].from_numpy(shape_funcs)
        coarse_basis[edge] = N

        # ### edge test sets
        dofs = tp.subproblem.V_to_L[edge]
        test_set[edge] = range_spaces[edge].from_numpy(M.dofs(dofs))
        # subtract coarse scale part
        test_cvals = test_set[edge].dofs(edge_boundary_dofs[edge])
        test_set[edge] -= N.lincomb(test_cvals)

        # ### initialize maxnorm
        if edge in active_edges:
            maxnorm = np.append(maxnorm, np.inf)
            # ### pod bases
            pod_bases[edge] = range_spaces[edge].empty()
        else:
            maxnorm = np.append(maxnorm, 0.0)

    timer.stop()
    logger.debug(f"Preparing stuff took t={timer.dt}s.")

    # NOTE tp.source is the full space, while the source product
    # is of lower dimension
    num_source_dofs = len(tp._bc_dofs_gamma_out)
    testfail = np.array(
            [failure_tolerance / min(num_source_dofs, space.dim)
             for space in range_spaces.values()])
    testlimit = (
        np.sqrt(2.0 * lambda_min) * erfinv(
            testfail ** (1.0 / num_testvecs)) * error_tol
    )

    logger.info(f"{lambda_min=}")
    logger.info(f"{testlimit=}")

    # initialization
    num_solves = 0
    rtol = 5e-2
    Lcorr = sampling_options.get("correlation_length")
    distance = sampling_options.get("distance")
    mean = sampling_options.get("mean")
    D = diags(mean)
    Σ, neig = get_sigma_neig(distance, Lcorr, D, rtol=rtol)

    while np.any(maxnorm > testlimit):

        if num_solves >= neig:
            # decrease correlation length
            Lcorr /= 2
            Σ, neig = get_sigma_neig(distance, Lcorr, D, rtol=rtol)

        v = tp.generate_random_boundary_data(
                1, distribution=distribution,
                random_state=random_state, mean=mean, cov=Σ.toarray()
                )

        U = tp.solve(v)
        num_solves += 1

        target_edges = edges[maxnorm > testlimit]
        for edge in target_edges:
            B = pod_bases[edge]
            edge_space = range_spaces[edge]
            # restrict the training sample to the edge
            Udofs = edge_space.from_numpy(U.dofs(tp.subproblem.V_to_L[edge]))
            coarse_values = Udofs.dofs(edge_boundary_dofs[edge])
            U_fine = Udofs - coarse_basis[edge].lincomb(coarse_values)

            # extend pod basis
            extend_basis(U_fine, B, product=range_products[edge],
                         method="pod", pod_modes=1)

            # orthonormalize test set wrt pod basis
            M = test_set[edge]
            M -= B.lincomb(B.inner(M, range_products[edge]).T)

            norm = M.norm(range_products[edge])
            maxnorm[edge_index_map[edge]] = np.max(norm)

        logger.info(f"{maxnorm=}")

    return pod_bases, range_products, num_solves
