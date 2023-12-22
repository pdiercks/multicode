import pytest
import numpy as np
from multi.sampling import create_random_values


@pytest.mark.parametrize("distribution", ["uniform", "normal", "multivariate_normal"])
def test(distribution):
    dim = 5
    options = {}
    if distribution == "multivariate_normal":
        options["mean"] = np.zeros(dim)
        options["cov"] = np.eye(dim)
    u = create_random_values((1, dim), distribution, **options)
    v = create_random_values((1, dim), distribution, **options)
    assert u.shape == (1, dim)
    assert not np.allclose(u, v)

    with pytest.raises(NotImplementedError):
        create_random_values((1, dim), distribution='not_supported')

    # raise value error if a key does not match distribution method
    kwargs = {}
    kwargs['notakey'] = 'value'
    with pytest.raises(ValueError):
        create_random_values((1, dim), distribution=distribution, **kwargs)

    # value error for wrong values
    if distribution == "uniform":
        with pytest.raises(ValueError):
            create_random_values((1, dim), distribution=distribution, low=1., high=0.)

    # value error if required value is not provided
    if distribution == "multivariate_normal":
        with pytest.raises(ValueError):
            create_random_values((2, dim), distribution='multivariate_normal', mean=None)
        with pytest.raises(ValueError):
            create_random_values((2, dim), distribution='multivariate_normal', mean=np.zeros(2), cov=None)

# python3 tests/test_sampling.py 0 # samples vectors and writes them
# python3 tests/test_sampling.py 1 # samples vectors and compares to first run
#
# def test(iter, outfile):
#     """test sampling behaviour"""
#     dim = 10
#     distribution = 'normal'
#     kwargs = {'loc': 0., 'scale': 1.}
#
#
#     u = create_random_values((1, dim), distribution, **kwargs)
#     v = create_random_values((1, dim), distribution, **kwargs)
#     assert not np.allclose(u, v)
#
#     with new_rng(33):
#         w = create_random_values((1, dim), distribution, **kwargs)
#         x = create_random_values((1, dim), distribution, **kwargs)
#
#     assert not np.allclose(u, w)
#     assert not np.allclose(w, x)
#
#     if iter < 1:
#         np.savez(outfile, u, v, w, x)
#     else:
#         # second pass should produce same random vectors
#         data = np.load(outfile)
#         assert np.allclose(u, data["arr_0"])
#         assert np.allclose(v, data["arr_1"])
#         assert np.allclose(w, data["arr_2"])
#         assert np.allclose(x, data["arr_3"])
#
#
# if __name__ == "__main__":
#     import sys
#     try:
#         args = sys.argv[1:]
#         iter = int(args[0])
#     except:
#         iter = 0
#     tf = Path(__file__).parent / "mystuff.npz"
#     test(iter, tf.as_posix())
#     if iter > 0:
#         tf.unlink()
