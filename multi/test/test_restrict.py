import numpy as np
from multi.interpolation import make_mapping
from multi.extension import restrict
from multi.boundary import within_range
import dolfinx
from mpi4py import MPI


def test_restrict_subdomain():
    num_cells = 10
    degree = 1

    # rectangle
    domain = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [2.0, 2.0]],
        [num_cells, num_cells],
        dolfinx.mesh.CellType.quadrilateral,
    )
    V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", degree))
    u = dolfinx.fem.Function(V)
    u.x.set(1.0)

    subdomain = within_range([0.5, 0.5, 0.], [1.5, 1.5, 0.])
    # h = 0.2, unit length = 1.
    # --> 5 elements --> 36 nodes --> 72 dofs

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
    V = dolfinx.fem.FunctionSpace(domain, ("CG", degree))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: 1. + x[0] ** 2 + 2 * x[1] ** 2)

    # bottom edge
    interval = dolfinx.mesh.create_interval(MPI.COMM_WORLD, num_cells, [0.0, 2.0])
    L = dolfinx.fem.FunctionSpace(interval, ("CG", degree))
    dofs = make_mapping(L, V)

    def bottom(x):
        return np.isclose(x[1], 0.0)

    # restriction
    r = restrict(u, bottom, 1, boundary=True)
    assert np.allclose(u.x.array[dofs], r)



if __name__ == "__main__":
    test_restrict_bottom()
