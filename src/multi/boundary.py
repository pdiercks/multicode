"""helpers to define boundaries"""

import numpy as np
from multi.common import to_floats

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
    """mark the domain within range

    Note: best used together with dolfinx.mesh.locate_entities_boundary
    and topological definition of the Dirichlet bc
    """
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


def show_marked(domain, marker):
    from dolfinx import fem
    from basix.ufl import element
    import matplotlib.pyplot as plt

    fe = element("Lagrange", domain.basix_cell(), 1, shape=())
    V = fem.functionspace(domain, fe)
    dofs = fem.locate_dofs_geometrical(V, marker)
    u = fem.Function(V)
    bc = fem.dirichletbc(u, dofs)
    x_dofs = V.tabulate_dof_coordinates()
    x_dofs = x_dofs[:, :2]
    marked = x_dofs[bc._cpp_object.dof_indices()[0]]

    plt.figure(1)
    x, y = x_dofs.T
    plt.scatter(x, y, facecolors="none", edgecolors="k", marker="o")
    xx, yy = marked.T
    plt.scatter(xx, yy, facecolors="r", edgecolors="none", marker="o")
    plt.show()
