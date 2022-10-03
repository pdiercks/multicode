"""test make mapping"""

import dolfinx
import numpy as np
from mpi4py import MPI
from multi.misc import make_mapping


def test_function_space():
    num_cells = 13
    degree = 2

    # rectangle
    domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [2.0, 2.0]],
        [num_cells, num_cells],
        dolfinx.mesh.CellType.quadrilateral,
    )
    V = dolfinx.fem.FunctionSpace(domain, ("CG", degree))

    # bottom edge
    interval = dolfinx.mesh.create_interval(MPI.COMM_WORLD, num_cells, [0.0, 2.0])
    L = dolfinx.fem.FunctionSpace(interval, ("CG", degree))
    x_dofs_L = L.tabulate_dof_coordinates()
    dofs = make_mapping(V, x_dofs_L.T)

    x_dofs_V = V.tabulate_dof_coordinates()
    assert dofs.size == L.dofmap.index_map.size_global
    assert np.allclose(x_dofs_L, x_dofs_V[dofs])


def test_vector_function_space():
    num_cells = 13
    degree = 2

    def xdofs_VectorFunctionSpace(V):
        bs = V.dofmap.bs
        x = V.tabulate_dof_coordinates()
        x_dofs = np.repeat(x, bs).reshape(x.shape[0] * bs, x.shape[1])
        return x_dofs

    # rectangle
    rectangle = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [2.0, 2.0]],
        [num_cells, num_cells],
        dolfinx.mesh.CellType.quadrilateral,
    )
    V = dolfinx.fem.VectorFunctionSpace(rectangle, ("CG", degree))
    x_dofs_V = xdofs_VectorFunctionSpace(V)

    # bottom edge
    interval = dolfinx.mesh.create_interval(MPI.COMM_WORLD, num_cells, [0.0, 2.0])
    L = dolfinx.fem.VectorFunctionSpace(interval, ("CG", degree), dim=2)
    bottom_points = L.tabulate_dof_coordinates()  # each coordinate only once
    dofs = make_mapping(V, bottom_points.T)

    x_dofs_L = xdofs_VectorFunctionSpace(L)
    assert dofs.size == L.dofmap.index_map.size_global * L.dofmap.bs
    assert np.allclose(x_dofs_L, x_dofs_V[dofs])


if __name__ == "__main__":
    test_function_space()
    test_vector_function_space()
