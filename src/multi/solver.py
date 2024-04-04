from contextlib import ExitStack
from typing import Optional
import numpy as np

from dolfinx import fem, la
from petsc4py import PETSc


def build_nullspace(V: fem.FunctionSpace, gdim: Optional[int] = 2) -> list[PETSc.Vec]: # type: ignore
    """Builds PETSc nullspace for elasticity problem.

    Args:
        V: The FE space.
        gdim: The geometric dimension.

    Returns:
        ns: The null space of a linear elastic problem.

    Note:
        `gdim=1` is not supported.

    """

    if gdim not in (2, 3):
        raise NotImplementedError

    n_trans = gdim
    if gdim == 3:
        n_rot = 3
    else:
        n_rot = 1

    ns_dim = n_trans + n_rot

    # Create list of vectors for building nullspace
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    ns = [la.create_petsc_vector(index_map, bs) for _ in range(ns_dim)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace (x, y and z dofs)
        dofs = [V.sub(i).dofmap.list for i in range(n_trans)]

        # Build the three translational rigid body modes
        for i in range(n_trans):
            basis[i][dofs[i]] = 1.0

        # Build the three rotational rigid body modes
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        if gdim == 3:
            basis[3][dofs[0]] = -x1
            basis[3][dofs[1]] = x0
            basis[4][dofs[0]] = x2
            basis[4][dofs[2]] = -x0
            basis[5][dofs[2]] = x1
            basis[5][dofs[1]] = -x2
        else:
            basis[2][dofs[0]] = -x1
            basis[2][dofs[1]] = x0

    return ns
