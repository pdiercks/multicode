"""test hierarchical shape functions"""
import numpy as np
from multi.shapes import _get_hierarchical_shape_fun_expr, mapping


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


def test():
    x = np.linspace(0, 1, num=51)
    # map phyiscal to reference coordinates
    xi = mapping(x, x.min(), x.max())
    equal = []
    for degree in range(2, 4):
        f_ex = analytical(xi, degree)
        shape_fun = _get_hierarchical_shape_fun_expr(degree)
        f = shape_fun(xi)
        equal.append(np.allclose(f_ex, f))
    assert np.all(equal)


if __name__ == "__main__":
    test()
