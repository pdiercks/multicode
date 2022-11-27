"""miscellaneous helpers"""

import numpy as np


def x_dofs_VectorFunctionSpace(V):
    bs = V.dofmap.bs
    x = V.tabulate_dof_coordinates()
    x_dofs = np.repeat(x, repeats=bs, axis=0)
    return x_dofs


def locate_dofs(x_dofs, X, gdim=2, s_=np.s_[:], tol=1e-9):
    """returns dofs at coordinates X

    Parameters
    ----------
    x_dofs : np.ndarray
        An array containing the coordinates of the DoFs of the FE space.
        Most likely the return value of V.tabulate_dof_coordinates().
    X : list, np.ndarray
        A list of points, where each point is given as list of len(gdim).
    gdim : int, optional
        The geometrical dimension of the domain.
    s_ : slice, optional
        Return slice of the dofs at each point.
    tol : float, optional
        Tolerance used to find coordinate.

    Returns
    -------
    dofs : np.ndarray
        DoFs at given coordinates.
    """
    if isinstance(X, list):
        X = np.array(X).reshape(len(X), gdim)
    elif isinstance(X, np.ndarray):
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
