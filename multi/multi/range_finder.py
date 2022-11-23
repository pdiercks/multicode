import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator

from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.special import erfinv

from multi.sampling import correlation_function


def nested_adaptive_rrf(
    transfer_problem,
    random_state,
    correlation_length=None,
    source_product=None,
    range_product=None,
    error_tol=1e-4,
    failure_tolerance=1e-15,
    num_testvecs=20,
    lambda_min=None,
    **kwargs,
):
    r"""Adaptive randomized range approximation of `A`.
    """

    logger = getLogger("multi.range_finder.adaptive_rrf", level="DEBUG")
    tp = transfer_problem

    assert source_product is None or isinstance(source_product, Operator)
    assert range_product is None or isinstance(range_product, Operator)

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

    # NOTE tp.source is the full space, while the source product is of lower dimension
    num_source_dofs = len(tp._bc_dofs_gamma_out)
    testfail = failure_tolerance / min(num_source_dofs, tp.range.dim)
    testlimit = (
        np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * error_tol
    )
    maxnorm = np.inf

    distance = kwargs.get("distance")
    mean = kwargs.get("mean")
    D = np.diag(mean)

    def compute_covariance(distance, lc, rtol=0.1):
        Σ_exp = correlation_function(distance, lc, function_type="exponential")
        Σ = np.dot(D, np.dot(Σ_exp, D))
        eigvals = eigh(Σ, eigvals_only=True, turbo=True)
        eigvals = eigvals[::-1]
        eigvals /= eigvals[0]
        mask = eigvals > rtol
        n = eigvals[mask].size
        return Σ, n


    # initial correlation length and covariance
    lc = correlation_length
    Σ, n_eigvals = compute_covariance(distance, lc)
    options = {"mean": mean, "cov": Σ}

    # global test set
    R = tp.generate_random_boundary_data(
            count=n_eigvals, distribution="multivariate_normal", random_state=random_state, **options
            )
    M = tp.solve(R)

    logger.info(f"{lambda_min=}")
    logger.info(f"{testlimit=}")

    B = tp.range.empty()
    U = tp.range.empty()

    generated_with = []

    # how to compute/choose testlimit if M is build iteratively?
    while maxnorm > testlimit:
        basis_length = len(B)

        logger.debug(f"{basis_length=}")
        logger.debug(f"{lc=}")
        logger.debug(f"{n_eigvals=}")

        if not basis_length < n_eigvals:
            # decrease correlation length and update covariance
            lc /= 2
            Σ, n_eigvals = compute_covariance(distance, lc)
            options = {"mean": mean, "cov": Σ}
            # extend testing set 
            M.append(tp.solve(tp.generate_random_boundary_data(
                count=n_eigvals, distribution="multivariate_normal", random_state=random_state, **options)))
            n_eigvals += basis_length

        generated_with.append(lc)
        v = tp.generate_random_boundary_data(1, distribution="multivariate_normal", random_state=random_state, **options)

        u = tp.solve(v)
        U.append(u)
        B.append(u)

        gram_schmidt(
            B,
            range_product,
            atol=0,
            rtol=0,
            offset=basis_length,
            copy=False,
        )
        # requires B to be orthonormal wrt range_product
        M -= B.lincomb(B.inner(M, range_product).T)

        norm = M.norm(range_product)
        if any(np.isnan(norm)):
            breakpoint()
        maxnorm = np.max(norm)
        logger.info(f"{maxnorm=}")
    logger.debug(f"{generated_with=}")

    return U


# modified version of pymor.algorithms.rand_la.adaptive_rrf
def adaptive_rrf(
    transfer_problem,
    random_state,
    distribution,
    source_product=None,
    range_product=None,
    error_tol=1e-4,
    failure_tolerance=1e-15,
    num_testvecs=20,
    lambda_min=None,
    **kwargs,
):
    r"""Adaptive randomized range approximation of `A`.
    This is an implementation of Algorithm 1 in [BS18]_.

    Given the |Operator| `A`, the return value of this method is the |VectorArray|
    `B` with the property

    .. math::
        \Vert A - P_{span(B)} A \Vert \leq tol

    with a failure probability smaller than `failure_tolerance`, where the norm denotes the
    operator norm. The inner product of the range of `A` is given by `range_product` and
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
    source_product
        Inner product |Operator| of the source of A.
    range_product
        Inner product |Operator| of the range of A.
    error_tol
        Error tolerance for the algorithm.
    failure_tolerance
        Maximum failure probability.
    num_testvecs
        Number of test vectors.
    lambda_min
        The smallest eigenvalue of source_product.
        If `None`, the smallest eigenvalue is computed using scipy.
    kwargs
        Optional keyword arguments for the generation of
        random samples (training data).
        see `_create_random_values`.

    Returns
    -------
     U
        |VectorArray| which contains the (non-orthonormal) solutions, whose
        span approximates the range of A.
    """

    logger = getLogger("multi.range_finder.adaptive_rrf")
    tp = transfer_problem

    assert source_product is None or isinstance(source_product, Operator)
    assert range_product is None or isinstance(range_product, Operator)

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

    # NOTE tp.source is the full space, while the source product is of lower dimension
    num_source_dofs = len(tp._bc_dofs_gamma_out)
    testfail = failure_tolerance / min(num_source_dofs, tp.range.dim)
    testlimit = (
        np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * error_tol
    )
    maxnorm = np.inf

    # set of test vectors
    if distribution == "multivariate_normal":
        # FIXME what is a good testing set in this case?
        # assumption: only test against macro state such that basis
        # approximates exactly this state well
        mean = kwargs.get("mean")
        if mean is None:
            raise ValueError
        R = tp.generate_boundary_data(mean.reshape(1, -1))
        if num_testvecs > 1:
            R.append(tp.generate_random_boundary_data(count=num_testvecs-1, distribution=distribution, random_state=random_state, **kwargs))
    else:
        R = tp.generate_random_boundary_data(
                count=num_testvecs, distribution=distribution, random_state=random_state
                )
    M = tp.solve(R)

    logger.info(f"{lambda_min=}")
    logger.info(f"{testlimit=}")

    B = tp.range.empty()
    U = tp.range.empty()
    while maxnorm > testlimit:
        basis_length = len(B)
        v = tp.generate_random_boundary_data(1, distribution, random_state, **kwargs)
        u = tp.solve(v)
        U.append(u)
        B.append(u)

        gram_schmidt(
            B,
            range_product,
            atol=0,
            rtol=0,
            offset=basis_length,
            copy=False,
        )
        # requires B to be orthonormal wrt range_product
        M -= B.lincomb(B.inner(M, range_product).T)

        norm = M.norm(range_product)
        if any(np.isnan(norm)):
            breakpoint()
        maxnorm = np.max(norm)
        logger.info(f"{maxnorm=}")

    return U
