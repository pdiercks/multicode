"""range finder algorithm

use same function to compute snapshots for the
construction of the empirical basis for
    (a) random training samples from a gaussian normal distribution
    (b) random training samples from a multivariate normal distribution
        with the macroscopic displacement as mean


inputs:
    logger
    tranfer_problem (weak form, bcs, Γ_out)
    random_state
    source_product <-- transfer_problem
    range_product <-- transfer_problem
    tol
    failure_tolerance
    num_testvecs
    lambda_min
    optional: u_macro
    optional: correlation length

returns:
    VectorArray (snapshots)

random_state <-- pymor.tools.random.get_random_state (returns numpy R state)
(a) --> random_state.normal(loc=0.0, scale=1.0, size=None)
(b) --> random_state.multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)


pymor.vectorarrays.interface._create_random_values might have
benefit of controlling optional kwargs (like `loc` and `scale` in case of 
random_state.normal for example)

Questions:
----------
1. in case of (b): how are the test vectors generated?
2. just compute orthonormal basis and decompose afterwards? or does this somehow lessen the quality of the POD edge basis?

"""

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.operators.interface import Operator

from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.special import erfinv


# modified version of pymor.algorithms.rand_la.adaptive_rrf
def adaptive_rrf(
    logger,
    transfer_problem,
    random_state,
    distribution,
    source_product=None,
    range_product=None,
    tol=1e-4,
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
    logger
        The logger used by the main program.
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
    tol
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
     B
        |VectorArray| which contains the basis, whose span approximates the range of A.
    """

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
        np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * tol
    )
    maxnorm = np.inf


    # set of test vectors
    # FIXME in general don't know what a good testing set is, such that
    # space is tailored to `u_macro` and gives good approx with few functions
    # M only controls how many basis functions are added,
    # add only functions which result in big maxnorm reduction???
    # test_vector_data = random_state.multivariate_normal(u_macro, Σ_scaled, size=num_testvecs)
    # R = problem.generate_boundary_data(test_vector_data)

    # TODO implement transfer_problem.generate_random_boundary_data(
    #                   self, count, distribution, random_state, seed, kwargs
    #               )
    # instead of tp.source use tp._bc_dofs_gamma_out
    # Q: why do I need random state and seed?
    # wanted behaviour: use the same random_state for one while loop

    # TODO at some point have to create random_state using seed or
    # get the default random state ...
    # in pymor this is done with source.random()
    # --> thus create random state in tp.generate_random_boundary_data

    # TODO it is important to be able to set the seed
    # for a specific realization of the empirical basis
    # this can be done by e.g.
    # rs = pymor.get_random_state(seed=222)
    # B = adaptive_rrf(logger, prob, rs, ...)

    # set of test vectors
    if distribution == "multivariate_normal":
        mean = kwargs.get("mean")
        if mean is None:
            raise ValueError
        R = tp.generate_boundary_data(mean.reshape(1, -1))
    else:
        R = tp.generate_random_boundary_data(
                count=num_testvecs, distribution=distribution, random_state=random_state
                )
    M = tp.solve(R)

    logger.info(f"{lambda_min=}")
    logger.info(f"{testlimit=}")

    B = tp.range.empty()
    while maxnorm > testlimit:
        basis_length = len(B)
        v = tp.generate_random_boundary_data(1, distribution, random_state, **kwargs)
        B.append(tp.solve(v))

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

    return B
