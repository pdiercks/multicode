"""helpers to define boundaries"""
# import dolfin as df
import numpy as np
from multi.common import to_floats

# from fenics_helpers.boundary import plane_at

"""Design

old dolfin: 
    here on needed a SubDomain object that defined the boundary geometrically.
    SubDomain could then be passed to DirichletBC.
    Therefore, fenics_helpers.boundary was used to conveniently define boundaries
    geometrically (would return a SubDomain).

dolfinx:
    input to dirichletbc is now:
        1. (Function, array)
        2. ([Constant, array], array, FunctionSpace)
    The array are the boundary_dofs which are determined via `locate_dofs_topological` or
    `locate_dofs_geometrical`.

    Thus, multi.boundary could provide functions to:
        (a) define callables that define complex geometry as input to locate_dofs_geometrical.
        (b) define functions that compute entities of the mesh and pass this array to locate_dofs_topological.

    (b) might use dolfinx.mesh.locate_entities and dolfinx.mesh.locate_entities_boundary

    Args:
        mesh: dolfinx.mesh.Mesh
        dim: tdim of the entities
        marker: function that takes an array of points x and returns an array of booleans

    --> therefore, use of locate_dofs_topological again boils down to a geometrical description
    of the boundary to be defined. The only difference is the possibility to filter wrt the tdim.
    (this is not possible with locate_dofs_geometrical)

"""


def plane_at(coordinate, dim):
    """return callable that determines boundary geometrically

    Parameters
    ----------
    coordinate : float
    dim : str or int
    """
    if dim in ["x", "X"]:
        dim = 0
    if dim in ["y", "Y"]:
        dim = 1
    if dim in ["z", "Z"]:
        dim = 2

    assert dim in (0, 1, 2)

    def boundary(x):
        return np.isclose(x[dim], coordinate)

    return boundary


def within_range(start, end, tol=1e-6):
    start = to_floats(start)
    end = to_floats(end)

    # adjust the values such that start < end for all dimensions
    assert len(start) == 3
    assert len(start) == len(end)
    for i in range(len(start)):
        if start[i] > end[i]:
            start[i], end[i] = end[i], start[i]

    def boundary(x):
        def in_range(i):
            return np.logical_and(x[i] >= start[i] - tol, x[i] <= end[i] + tol)

        xy = np.logical_and(in_range(0), in_range(1))
        return np.logical_and(xy, in_range(2))

    return boundary


def point_at(coord):
    p = to_floats(coord)
    assert len(p) == 3

    def boundary(x):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], p[0]), np.isclose(x[1], p[1])),
            np.isclose(x[2], p[2]),
        )

    return boundary


# TODO
# def l_shaped_boundary_factory(index, x_range, y_range, tol=1e-6):
#     r"""return L-shaped boundary of a rectangular domain opposite
#     to the domain of interest as shown below.

#         •-----•-----•
#         |  1  |  3  |
#         |     |     |
#         •-----•-----•
#         |  0  |  2  |
#         |     |     |
#         •-----•-----•


#     Parameters
#     ----------
#     index : int
#         Domain of interest index following above definition.
#     x_range : tuple of float
#         Range of the x-coordinate of the domain.
#     y_range : tuple of float
#         Range of the y-coordinate of the domain.
#     tol : float, optional
#         Tolerance used to exclude points on ∂Ω_gl (Γ = ∂Ω \ ∂Ω_gl).

#     Returns
#     -------
#     gamma : boundary function

#     """
#     bottom = plane_at(y_range[0], "y")
#     right = plane_at(x_range[1], "x")
#     top = plane_at(y_range[1], "y")
#     left = plane_at(x_range[0], "x")
#     if index == 3:

#         def gamma(x, on_boundary):
#             c1 = left.inside(x, on_boundary) and x[1] < y_range[1] - tol
#             c2 = bottom.inside(x, on_boundary) and x[0] < x_range[1] - tol
#             return c1 or c2

#     elif index == 2:

#         def gamma(x, on_boundary):
#             c1 = left.inside(x, on_boundary) and x[1] > y_range[0] + tol
#             c2 = top.inside(x, on_boundary) and x[0] < x_range[1] - tol
#             return c1 or c2

#     elif index == 1:

#         def gamma(x, on_boundary):
#             c1 = bottom.inside(x, on_boundary) and x[0] > x_range[0] + tol
#             c2 = right.inside(x, on_boundary) and x[1] < y_range[1] - tol
#             return c1 or c2

#     elif index == 0:

#         def gamma(x, on_boundary):
#             c1 = top.inside(x, on_boundary) and x[0] > x_range[0] + tol
#             c2 = right.inside(x, on_boundary) and x[1] > y_range[0] + tol
#             return c1 or c2

#     else:
#         raise NotImplementedError

#     return gamma


# TODO
# def u_shaped_boundary_factory(dim, coord, tol=1e-9):
#     """return U shaped boundary function where the plane given
#     by `dim` and `coord` is excluded"""

#     if dim in ["x", "X"]:
#         dim = 0
#     if dim in ["y", "Y"]:
#         dim = 1

#     assert dim in [0, 1]

#     def boundary(x, on_boundary):
#         if df.near(x[dim], coord, tol):
#             return False
#         else:
#             return on_boundary

#     return boundary
