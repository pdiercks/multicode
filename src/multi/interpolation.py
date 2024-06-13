from dolfinx import fem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import numpy as np
import numpy.typing as npt


# this code is part of the fenicsx-tutorial written by Jørgen S. Dokken
# see https://jorgensd.github.io/dolfinx-tutorial/chapter1/membrane_code.html#making-curve-plots-throughout-the-domain
def interpolate(u: fem.Function, points: npt.NDArray[np.float64]):
    """Evaluates u at points.

    Args:
        u: The function to evaluate.
        points: The points at which to evaluate.

    """
    V = u.function_space
    domain = V.mesh
    tree = bb_tree(domain, domain.topology.dim)

    cells = []
    points_on_proc = []

    if not points.shape[1] == 3:
        points = points.T
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(
        domain, cell_candidates, points
    )
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc, cells)
    return u_values


def make_mapping(subspace, superspace, padding=1e-14, check=True):
    """Computes dof mapping from superspace to subspace.

    Returns
    -------
    dofs : np.ndarray
        The dofs of `superspace` corresponding to dofs of `subspace`.
    """
    u = fem.Function(superspace)
    ndofs = superspace.dofmap.index_map.size_local * superspace.dofmap.bs
    u.x.array[:] = np.arange(ndofs, dtype=np.int32)

    f = fem.Function(subspace)

    interp_data = fem.create_nonmatching_meshes_interpolation_data(
            f.function_space.mesh, f.function_space.element, u.function_space.mesh,
            padding=padding)
    f.interpolate(u, nmm_interpolation_data=interp_data)
    f.x.scatter_forward()
    dofs = (f.x.array[:] + 0.5).astype(np.int32).flatten()

    if check:
        _, counts = np.unique(dofs, return_counts=True)
        if not np.isclose(np.sum(counts), dofs.size):
            raise ValueError("Something went wrong! Dofmap is not unique!")
    return dofs


def build_dof_map(V, W, padding=1e-14):
    """Returns DOFs of V corresponding to DOFs of W.

    Args:
        V: The parent or super FE space.
        W: The child or sub FE space.
    """

    u = fem.Function(V)
    uvec = u.x.petsc_vec

    w = fem.Function(W)
    wvec = w.x.petsc_vec

    interp_data = fem.create_nonmatching_meshes_interpolation_data(
            V.mesh,
            V.element,
            W.mesh,
            padding=padding)

    ndofs = W.dofmap.bs * W.dofmap.index_map.size_local
    dofs = np.arange(ndofs, dtype=np.int32)

    super = []
    for dof in dofs:
        wvec.zeroEntries()
        wvec.array[dof] = 1.0
        w.x.scatter_forward()

        uvec.zeroEntries()
        u.interpolate(w, nmm_interpolation_data=interp_data)
        u.x.scatter_forward()

        u_array = uvec.array
        if not np.all(np.logical_or(np.abs(u_array) < 1e-10, np.abs(u_array - 1.0) < 1e-10)):
            raise NotImplementedError
        u_dof = np.where(np.abs(u_array - 1.0) < 1e-10)[0]
        if not len(u_dof) == 1:
            raise NotImplementedError
        super.append(u_dof[0])
    super = np.array(super, dtype=np.int32)
    assert len(set(super)) == len(set(dofs))
    return super
