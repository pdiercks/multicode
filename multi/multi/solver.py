import dolfinx
from contextlib import ExitStack
import numpy as np
# from petsc4py import PETSc
from multi.product import InnerProduct
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator


def build_nullspace(V, product=None, gdim=2):
    """Build PETSc nullspace for 3D elasticity"""
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
    ns = [dolfinx.la.create_petsc_vector(index_map, bs) for i in range(ns_dim)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace (x, y and z dofs)
        dofs = [V.sub(i).dofmap.list.array for i in range(n_trans)]

        # Build the three translational rigid body modes
        for i in range(n_trans):
            basis[i][dofs[i]] = 1.0

        # Build the three rotational rigid body modes
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        # TODO this could be improved
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

    # Orthonormalise the basis using pymor
    source = FenicsxVectorSpace(V)
    B = source.make_array(ns)

    inner_product = InnerProduct(V, product=product)
    matrix = inner_product.assemble_matrix()
    if matrix is not None:
        operator = FenicsxMatrixOperator(matrix, V, V)
    else:
        operator = None
    gram_schmidt(B, operator, atol=0.0, rtol=0.0, copy=False)

    # return value?
    # --> to remove the kernel I need the pymor VectorArray
    # --> for use in an iterative solver need the vectors `ns`
    # return PETSc.NullSpace().create(vectors=ns)
    return B
