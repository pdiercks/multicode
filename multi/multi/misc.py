"""miscellaneous helpers"""

import dolfin as df
import numpy as np
from pymor.vectorarrays.numpy import NumpyVectorArray


def make_mapping(sub_space, super_space):
    """get map from sub to super space

    Parameters
    ----------
    sub_space
        A dolfin.FunctionSpace
    super_space
        A dolfin.FunctionSpace

    Returns
    -------
    The dofs of super_space located at entities in sub_space.

    Note: This only works for conforming meshes.
    """
    f = df.Function(super_space)
    f.vector().set_local(super_space.dofmap().dofs())
    If = df.interpolate(f, sub_space)
    return (If.vector().get_local() + 0.5).astype(int)


def read_bases(*args, modes_per_edge=None):
    """read bases from args where each arg is a tuple of FilePath and
    tuple of string defining edge(s) for which to load basis functions from FilePath
    e.g. arg=("PathToFile", ("b", "r")) meaning
        load basis functions for bottom and right from "PathToFile"

    """
    bases = []
    edges = []
    for basis, edge_set in args:
        bases.append(np.load(basis))
        edges.append(list(edge_set))

    # check coarse scale basis phi and all edges are defined
    alle = [s for sub_set in edges for s in sub_set]
    assert not len(set(alle).difference(["phi", "b", "r", "t", "l"]))

    R = []
    Nmodes = []

    # append coarse scale basis functions
    for i, edge_set in enumerate(edges):
        if "phi" in edge_set:
            R.append(bases[i]["phi"])
            edge_set.remove("phi")

    # determine max number of modes per edge
    max_modes = []
    for basis, edge_set in zip(bases, edges):
        if edge_set:
            max_modes.append(max([len(basis[e]) for e in edge_set]))
    max_modes = max(max_modes)
    if modes_per_edge is not None:
        m = int(modes_per_edge)
    else:
        m = max_modes

    # append fine scale basis functions
    for key in ["b", "r", "t", "l"]:
        for basis, edge_set in zip(bases, edges):
            if key in edge_set:
                rb = basis[key][:m]
                Nmodes.append(rb.shape[0])
                if rb.shape[0] < m:
                    # add zero dummy modes (in case of dirichlet or neumann boundary)
                    # such that rb.shape[0] == m
                    rb = np.vstack((rb, np.zeros((m - rb.shape[0], rb.shape[1]))))
                R.append(rb)
                break

    return np.vstack(R), tuple(Nmodes)


def select_modes(basis, modes_per_edge, max_modes):
    """select modes according to multi.DofMap

    Parameters
    ----------
    basis : np.ndarray
        The multiscale basis used.
    modes_per_edge : int
        Number of modes per edge.
    max_modes : int
        Maximum number of modes per edge.

    Returns
    -------
    basis : np.ndarray
        Subset of the full basis.

    """

    offset = 0
    coarse = [i for i in range(8)]
    offset += len(coarse)
    bottom = [offset + i for i in range(modes_per_edge)]
    offset += max_modes
    right = [offset + i for i in range(modes_per_edge)]
    offset += max_modes
    top = [offset + i for i in range(modes_per_edge)]
    offset += max_modes
    left = [offset + i for i in range(modes_per_edge)]
    ind = coarse + bottom + right + top + left
    return basis[ind]


def set_values(U, dofs, values):
    """set ``dofs`` entries of all vectors in VectorArray U to ``values``"""
    assert isinstance(U, NumpyVectorArray)
    # unfortunately, I cannot figure out how to achieve the same
    # for ListVectorArray of FenicsVectors
    array = U.to_numpy()
    array[:, dofs] = values


def restrict_to(domain, function):
    """restrict given function or list of functions to domain"""
    if isinstance(function, list):
        # assuming all functions are elements of V
        V = function[0].function_space()
        element = V.ufl_element()
        Vsub = df.FunctionSpace(domain.mesh, element)
        assert Vsub.dim() < V.dim()
        interp = []
        for f in function:
            If = df.interpolate(f, Vsub)
            interp.append(If)
        return interp
    else:
        V = function.function_space()
        element = V.ufl_element()
        Vsub = df.FunctionSpace(domain.mesh, element)
        assert Vsub.dim() < V.dim()
        return df.interpolate(function, Vsub)


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
