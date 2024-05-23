"""test make mapping"""

from mpi4py import MPI
import pytest
import dolfinx
import numpy as np
from basix.ufl import element
from multi.interpolation import make_mapping


@pytest.mark.parametrize("value_shape", [(), (2,)])
def test_function_space(value_shape):
    num_cells = 13
    degree = 2

    def get_xdofs(V):
        bs = V.dofmap.bs
        x = V.tabulate_dof_coordinates()
        x_dofs = np.repeat(x, repeats=bs, axis=0)
        return x_dofs

    # rectangle
    a = b = 1000.
    domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        np.array([[0.0, 0.0], [a, b]]),
        [num_cells, num_cells],
        dolfinx.mesh.CellType.quadrilateral,
    )
    quad = element("Lagrange", domain.basix_cell(), degree, shape=value_shape)
    V = dolfinx.fem.functionspace(domain, quad)

    # bottom edge
    interval = dolfinx.mesh.create_interval(MPI.COMM_WORLD, num_cells, [0.0, a])
    line = element("Lagrange", interval.basix_cell(), degree, shape=value_shape)
    L = dolfinx.fem.functionspace(interval, line)
    dofs = make_mapping(L, V)

    # V.tabulate_dof_coordinates() does not consider dofmap.bs
    # if dofmap.bs == 1, no problem
    x_dofs_V = get_xdofs(V)
    x_dofs_L = get_xdofs(L)

    assert dofs.size == L.dofmap.index_map.size_global * L.dofmap.bs
    assert np.allclose(x_dofs_L, x_dofs_V[dofs])


if __name__ == "__main__":
    value_shape = ()
    test_function_space(value_shape)
