import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist, squareform
from pymor.tools.random import get_rng


# modified version of pymor.vectorarrays.interface._create_random_values
# to be used with transfer_problem.generate_random_boundary_data
# shape = (count, source.dim)
def create_random_values(shape: tuple[int, int], distribution: str, **kwargs) -> npt.NDArray:
    """Draws random samples from given `distribution`.

    Args:
        shape: The shape of the returned array.
        distribution: The distribution to draw samples from.
        kwargs: Arguments to pass to distribution method.
        See e.g. `numpy.random.Generator.normal`.

    """
    supported = ['uniform', 'normal', 'multivariate_normal']
    if distribution not in supported:
        msg = f"Unsupported {distribution=}. It has to be one of {','.join(supported)}."
        raise NotImplementedError(msg)

    rng = get_rng()

    if distribution == 'uniform':
        if not kwargs.keys() <= {'low', 'high'}:
            raise ValueError
        low = kwargs.get('low', 0.)
        high = kwargs.get('high', 1.)
        if high <= low:
            raise ValueError
        return rng.uniform(low, high, shape)
    elif distribution == 'normal':
        if not kwargs.keys() <= {'loc', 'scale'}:
            raise ValueError
        loc = kwargs.get('loc', 0.)
        scale = kwargs.get('scale', 1.)
        return rng.normal(loc, scale, shape)
    elif distribution == 'multivariate_normal':
        if not kwargs.keys() <= {'mean', 'cov', 'size', 'check_valid', 'tol', 'method'}:
            raise ValueError
        mean = kwargs.get('mean')
        cov = kwargs.get('cov')
        size = kwargs.get('size', None) or shape[0]
        check_valid = kwargs.get('check_valid', 'warn')
        tol = kwargs.get('tol', 1e-8)
        method = kwargs.get('method', 'svd')
        if mean is None:
            raise ValueError
        if cov is None:
            raise ValueError
        return rng.multivariate_normal(mean, cov, size=size, check_valid=check_valid, tol=tol, method=method)
    else: # pragma: no cover
        assert False


def correlation_function(
    d, correlation_length, function_type="exponential", exponent=2
    ): # pragma: no cover
    """
    Statistical correlation between measurements made at two points that are at `d`
    units distance from each other.

    Args:
        d: distance(s) between points.
        correlation_length: `1/correlation_length` controls the strength of
            correlation between two points. `correlation_length = 0` => complete
            independence for all point pairs (`rho=0`). `correlation_length = Inf` =>
            full dependence for all point pairs (`rho=1`).
        function_type: name of the correlation function.
        exponent: exponent in the type="cauchy" function. The larger the exponent the
            less correlated two points are.

    Returns:
        rho: correlation coefficient for each element of `d`.
    """
    function_type = function_type.lower()

    if correlation_length < 0:
        raise ValueError("correlation_length must be a non-negative number.")
    if np.any(d < 0):
        raise ValueError("All elements of d must be non-negative numbers.")

    if correlation_length == 0:
        idx = d == 0
        rho = np.zeros(d.shape)
        rho[idx] = 1
    elif correlation_length == np.inf:
        rho = np.ones(d.shape)
    else:
        if function_type == "exponential":
            rho = np.exp(-d / correlation_length)
        elif function_type == "cauchy":
            rho = (1 + (d / correlation_length) ** 2) ** -exponent
        elif function_type == "gaussian":
            rho = np.exp(-((d / correlation_length) ** 2))
        else:
            raise ValueError(
                "Unknown function_type. It must be one of these: 'exponential', "
                "'cauchy', 'gaussian'."
            )
    return rho


def correlation_matrix(points, correlation_length, distance_metric="euclidean"): # pragma: no cover
    distance = squareform(pdist(points, metric=distance_metric))
    Σ = correlation_function(
        distance, correlation_length, function_type="exponential"
    )
    return Σ
