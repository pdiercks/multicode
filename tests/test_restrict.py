import numpy as np
from multi.interpolation import make_mapping
from multi.extension import restrict
from multi.boundary import within_range
import dolfinx
from mpi4py import MPI


def test_restrict_subdomain():
    num_cells = 10
    degree = 1
    length = 1.0

    # rectangle
    domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [length, length]],
        [num_cells, num_cells],
        dolfinx.mesh.CellType.quadrilateral,
    )
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", degree))
    u = dolfinx.fem.Function(V)
    u.x.set(1.0)

    subdomain = within_range([0.3, 0.3, 0.0], [0.8, 0.8, 0.0])
    # h = length / num_cells = 0.1
    # subdomain has is 0.5 width and height
    # therefore 5x5 = 25 cells
    # --> 25 elements --> 36 nodes --> 72 dofs

    # restriction
    r = restrict(u, subdomain, 2)
    assert np.isclose(np.sum(r), 36 * 2)


def test_restrict_bottom():
    num_cells = 8
    degree = 2

    # rectangle
    domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [2.0, 2.0]],
        [num_cells, num_cells],
        dolfinx.mesh.CellType.quadrilateral,
    )
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", degree))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: 1.0 + x[0] ** 2 + 2 * x[1] ** 2)

    # bottom edge
    interval = dolfinx.mesh.create_interval(MPI.COMM_WORLD, num_cells, [0.0, 2.0])
    L = dolfinx.fem.FunctionSpace(interval, ("Lagrange", degree))
    dofs = make_mapping(L, V)

    def bottom(x):
        return np.isclose(x[1], 0.0)

    # restriction
    r = restrict(u, bottom, 1, boundary=True)
    assert np.allclose(u.x.array[dofs], r)


if __name__ == "__main__":
    test_restrict_bottom()
    test_restrict_subdomain()
