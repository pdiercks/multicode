"""miscellaneous helpers"""

import dolfinx
import numpy as np
from pymor.vectorarrays.numpy import NumpyVectorArray


# FIXME workaround until dolfinx supports interpolation on different meshes
def make_mapping(V, points):
    domain = V.mesh
    bb_tree = dolfinx.geometry.BoundingBoxTree(domain, domain.topology.dim)

    u = dolfinx.fem.Function(V)
    ndofs = V.dofmap.index_map.size_global * V.dofmap.bs
    u.x.array[:] = np.arange(ndofs, dtype=np.intc)

    # TODO parallel: point_on_proc

    cells = []
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        domain, cell_candidates, points.T
    )
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            cells.append(colliding_cells.links(i)[0])
    dofs = (u.eval(points.T, cells) + 0.5).astype(np.intc)
    return dofs.reshape(
        dofs.size,
    )


# def make_mapping(sub_space, super_space):
#     """get map from sub to super space

#     Parameters
#     ----------
#     sub_space
#         A dolfin.FunctionSpace
#     super_space
#         A dolfin.FunctionSpace

#     Returns
#     -------
#     The dofs of super_space located at entities in sub_space.

#     Note: This only works for conforming meshes.
#     """
#     f = df.Function(super_space)
#     f.vector().set_local(super_space.dofmap().dofs())
#     If = df.interpolate(f, sub_space)
#     return (If.vector().get_local() + 0.5).astype(int)


def select_modes(basis, modes, max_modes):
    """select modes according to local dof order in multi.dofmap.DofMap

    Parameters
    ----------
    basis : np.ndarray
        The multiscale basis used.
    modes : int or list of int
        Number of modes per edge.
    max_modes : int or list of int
        Maximum number of modes per edge.

    Returns
    -------
    basis : np.ndarray
        Subset of the full basis.

    """
    if isinstance(max_modes, (int, np.integer)):
        max_modes = [max_modes] * 4
    if isinstance(modes, (int, np.integer)):
        modes = [modes] * 4

    # make sure that modes[edge] <= max_modes[edge]
    assert len(max_modes) == len(modes)
    for i in range(len(max_modes)):
        if modes[i] > max_modes[i]:
            modes[i] = max_modes[i]

    coarse = [i for i in range(8)]
    offset = len(coarse)
    mask = coarse
    for edge in range(4):
        mask += [offset + i for i in range(modes[edge])]
        offset += max_modes[edge]
    return basis[mask]


def set_values(U, dofs, values):
    """set ``dofs`` entries of all vectors in VectorArray U to ``values``"""
    if isinstance(U, NumpyVectorArray):
        # unfortunately, I cannot figure out how to achieve the same
        # for ListVectorArray of FenicsVectors
        array = U.to_numpy()
        array[:, dofs] = values
    else:
        space = U.space
        array = U.to_numpy()
        array[:, dofs] = values
        return space.from_numpy(array)


# TODO
# def restrict_to(domain, function):
#     """restrict given function or list of functions to domain"""
#     if isinstance(function, list):
#         # assuming all functions are elements of V
#         V = function[0].function_space()
#         element = V.ufl_element()
#         Vsub = df.FunctionSpace(domain.mesh, element)
#         assert Vsub.dim() < V.dim()
#         interp = []
#         for f in function:
#             If = df.interpolate(f, Vsub)
#             interp.append(If)
#         return interp
#     else:
#         V = function.function_space()
#         element = V.ufl_element()
#         Vsub = df.FunctionSpace(domain.mesh, element)
#         assert Vsub.dim() < V.dim()
#         return df.interpolate(function, Vsub)


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
