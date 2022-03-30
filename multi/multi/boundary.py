"""helpers to define boundaries"""
import dolfin as df
from fenics_helpers.boundary import plane_at


def l_shaped_boundary_factory(index, x_range, y_range, tol=1e-6):
    r"""return L-shaped boundary of a rectangular domain opposite
    to the domain of interest as shown below.

        •-----•-----•
        |  1  |  3  |
        |     |     |
        •-----•-----•
        |  0  |  2  |
        |     |     |
        •-----•-----•


    Parameters
    ----------
    index : int
        Domain of interest index following above definition.
    x_range : tuple of float
        Range of the x-coordinate of the domain.
    y_range : tuple of float
        Range of the y-coordinate of the domain.
    tol : float, optional
        Tolerance used to exclude points on ∂Ω_gl (Γ = ∂Ω \ ∂Ω_gl).

    Returns
    -------
    gamma : boundary function

    """
    bottom = plane_at(y_range[0], "y")
    right = plane_at(x_range[1], "x")
    top = plane_at(y_range[1], "y")
    left = plane_at(x_range[0], "x")
    if index == 3:

        def gamma(x, on_boundary):
            c1 = left.inside(x, on_boundary) and x[1] < y_range[1] - tol
            c2 = bottom.inside(x, on_boundary) and x[0] < x_range[1] - tol
            return c1 or c2

    elif index == 2:

        def gamma(x, on_boundary):
            c1 = left.inside(x, on_boundary) and x[1] > y_range[0] + tol
            c2 = top.inside(x, on_boundary) and x[0] < x_range[1] - tol
            return c1 or c2

    elif index == 1:

        def gamma(x, on_boundary):
            c1 = bottom.inside(x, on_boundary) and x[0] > x_range[0] + tol
            c2 = right.inside(x, on_boundary) and x[1] < y_range[1] - tol
            return c1 or c2

    elif index == 0:

        def gamma(x, on_boundary):
            c1 = top.inside(x, on_boundary) and x[0] > x_range[0] + tol
            c2 = right.inside(x, on_boundary) and x[1] > y_range[0] + tol
            return c1 or c2

    else:
        raise NotImplementedError

    return gamma


def u_shaped_boundary_factory(dim, coord, tol=1e-9):
    """return U shaped boundary function where the plane given
    by `dim` and `coord` is excluded"""

    if dim in ["x", "X"]:
        dim = 0
    if dim in ["y", "Y"]:
        dim = 1

    assert dim in [0, 1]

    def boundary(x, on_boundary):
        if df.near(x[dim], coord, tol):
            return False
        else:
            return on_boundary

    return boundary
