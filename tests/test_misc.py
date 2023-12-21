import pytest
from mpi4py import MPI
from dolfinx import fem, mesh
from basix.ufl import element
import numpy as np
from multi.misc import x_dofs_vectorspace, locate_dofs


def test():
    n = 6
    domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    ve = element("P", domain.basix_cell(), 2, shape=(2,))
    V = fem.functionspace(domain, ve)
    x_dofs = x_dofs_vectorspace(V)

    origin = np.array([0.0, 0.0, 0.0])
    dof_origin = locate_dofs(x_dofs, origin)

    with pytest.raises(NotImplementedError):
        # ndim == 3 is not supported
        origin = origin[np.newaxis, :]
        locate_dofs(x_dofs, origin[np.newaxis, :])

    with pytest.raises(IndexError):
        # test a point that is not a vertex of the grid
        locate_dofs(x_dofs, np.array([[0.05, 0.05, 0.0]]))

    vertices = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    dofs_verts_x = locate_dofs(x_dofs_vectorspace(V), vertices, s_=np.s_[0::2])
    dofs_verts_y = locate_dofs(x_dofs_vectorspace(V), vertices, s_=np.s_[1::2])

    all_dofs = np.hstack((dof_origin, dofs_verts_x, dofs_verts_y))
    assert np.isclose(all_dofs.size, np.sum(x_dofs[all_dofs]))


if __name__ == "__main__":
    test()
