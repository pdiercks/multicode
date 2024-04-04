"""miscellaneous helpers"""

import typing
import numpy as np
import numpy.typing as npt
from dolfinx.fem import FunctionSpace


def to_floats(x: typing.Union[typing.Iterable[int], typing.Iterable[float]]) -> list[float]:
    """Converts `x` to a 3d coordinate.

    Args:
        x: point coordinates at least 1D

    Returns:
        point described as list with x,y,z value
    """

    floats = []
    try:
        for v in x:
            floats.append(float(v))
        while len(floats) < 3:
            floats.append(0.0)
    except TypeError:
        raise TypeError(f"x={x} is not iterable!")

    return floats


def x_dofs_vectorspace(V: FunctionSpace) -> npt.NDArray:
    bs = V.dofmap.bs
    x = V.tabulate_dof_coordinates()
    x_dofs = np.repeat(x, repeats=bs, axis=0)
    return x_dofs


def locate_dofs(x_dofs: npt.NDArray, X: npt.NDArray, s_: slice = np.s_[:], tol: float = 1e-9) -> npt.NDArray:
    """Returns DOFs at coordinates X.

    Args:
        x_dofs: Coordinates of the DOFs of a FE space.
        X: Coordinates for which to determine DOFs.
        s_: Use slice to restrict which DOFs are returned.
        tol: Tolerance used to find coordinates.

    """
    if X.ndim == 1:
        X = X[np.newaxis, :]
    elif X.ndim > 2:
        raise NotImplementedError

    dofs = np.array([], int)
    for x in X:
        p = np.abs(x_dofs - x)
        v = np.where(np.all(p < tol, axis=1))[0]
        if v.size < 1:
            raise IndexError(f"The point {x} is not a vertex of the grid!")
        dofs = np.append(dofs, v[s_])

    return dofs
