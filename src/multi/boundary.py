"""Easy definition of boundaries."""

import typing

import dolfinx
import numpy as np
from basix.ufl import element
from multi.misc import to_floats


def plane_at(coordinate: float, dim: typing.Union[str, int]) -> typing.Callable:
    """Defines a plane where `x[dim]` equals `coordinate`.

    Args:
        coordinate: value
        dim: dimension

    Returns:
        Function defining the boundary.
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


def line_at(coordinates: list[float], dims: list[typing.Union[str, int]]) -> typing.Callable:
    """Defines a line where `x[dims[0]]` equals `coordinates[0]` and `x[dims[1]]` equals `coordinates[1]`.

    Args:
        coordinates: list of values
        dims: list of dimension

    Returns:
        Function defining the boundary.
    """

    assert len(coordinates) == 2
    assert len(dims) == 2

    # transform x,y,z str into integer
    for i, dim in enumerate(dims):
        if dim in ["x", "X"]:
            dims[i] = 0
        elif dim in ["y", "Y"]:
            dims[i] = 1
        elif dim in ["z", "Z"]:
            dims[i] = 2
        assert dims[i] in (0, 1, 2)

    assert dims[0] != dims[1]

    def boundary(x):
        return np.logical_and(
            np.isclose(x[dims[0]], coordinates[0]),
            np.isclose(x[dims[1]], coordinates[1]),
        )

    return boundary


def within_range(
    start: typing.Union[typing.Iterable[int], typing.Iterable[float]],
    end: typing.Union[typing.Iterable[int], typing.Iterable[float]],
    tol: float = 1e-6,
) -> typing.Callable:
    """Defines a range.

    It is best used together with `dolfinx.mesh.locate_entities_boundary`
    and topological definition of the Dirichlet BC, because the Callable
    will mark the whole range and not just the boundary.

    Args:
        start: The start point of the range.
        end: The end point of the range.

    Returns:
        function defining the boundary
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


def point_at(coord: typing.Union[typing.Iterable[int], typing.Iterable[float]]) -> typing.Callable:
    """Defines a point.

    Args:
        coord: points coordinates

    Returns:
        function defining the boundary
    """
    p = to_floats(coord)

    def boundary(x):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], p[0]), np.isclose(x[1], p[1])),
            np.isclose(x[2], p[2]),
        )

    return boundary


def show_marked(
    domain: dolfinx.mesh.Mesh,
    marker: typing.Union[typing.Callable, np.ndarray],
    entity_dim: typing.Optional[int] = None,
    filename: typing.Union[str, None] = None,
) -> None:  # pragma: no cover
    """Shows dof coordinates marked by `marker`.

    Notes:
      This is useful for debugging boundary conditions.
      Currently this only works for domains of topological
      dimension 2.

    Args:
        domain: The computational domain.
        marker: A function that takes an array of points ``x`` with shape
          ``(gdim, num_points)`` and returns an array of booleans of
          length ``num_points``, evaluating to ``True`` for entities whose
          degree-of-freedom should be returned. Instead of a function the
          entities whose degree-of-freedom should be returned can be passed
          directly. In the latter case `entity_dim` needs to be provided.
        entity_dim: The dimension of the entities associated with `marker`.
        filename: Save figure to this path.
          If None, the figure is shown (default).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required to show marked dofs.")

    tdim = domain.topology.dim
    if tdim in (1, 3):
        raise NotImplementedError(f"Not implemented for mesh of topological dimension {tdim=}.")

    fe = element("Lagrange", domain.basix_cell(), 1, shape=())
    V = dolfinx.fem.functionspace(domain, fe)
    if isinstance(marker, np.ndarray):
        assert entity_dim is not None
        dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim, marker)
    else:
        dofs = dolfinx.fem.locate_dofs_geometrical(V, marker)
    u = dolfinx.fem.Function(V)
    bc = dolfinx.fem.dirichletbc(u, dofs)
    x_dofs = V.tabulate_dof_coordinates()
    x_dofs = x_dofs[:, :2]
    marked = x_dofs[bc._cpp_object.dof_indices()[0]]

    plt.figure(1)
    x, y = x_dofs.T
    plt.scatter(x, y, facecolors="none", edgecolors="k", marker="o")
    xx, yy = marked.T
    plt.scatter(xx, yy, facecolors="r", edgecolors="none", marker="o")

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
