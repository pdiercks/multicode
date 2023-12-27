"""test hierarchical shape functions"""

import pytest
import numpy as np
from multi.shapes import get_hierarchical_shape_functions, mapping


def analytical(x, p):
    """
    _get_hierarchical_shape_fun_expr implements
    equation (8.61) in the book "The finite element
    method volume 1" by Zienkiewicz and Taylor
    """
    if p == 2:
        return x ** 2 - 1
    elif p == 3:
        return 2 * (x ** 3 - x)
    else:
        raise NotImplementedError


@pytest.mark.parametrize("shift", [0., 1.])
def test(shift):
    x = np.linspace(0, 1, num=51) + shift
    # map phyiscal to reference coordinates
    xi = mapping(x, x.min(), x.max())

    with pytest.raises(ValueError):
        get_hierarchical_shape_functions(x, 1)

    equal = []
    for degree in range(2, 4):
        f = get_hierarchical_shape_functions(x, degree, ncomp=1)[-1, :]
        f_ana = analytical(xi, degree).reshape(f.shape)
        equal.append(np.allclose(f_ana, f))
    assert np.all(equal)


if __name__ == "__main__":
    test(0.)
