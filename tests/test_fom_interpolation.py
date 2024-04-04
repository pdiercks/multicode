from mpi4py import MPI
import pytest
from dolfinx import fem, mesh
from basix.ufl import element
import numpy as np


@pytest.mark.parametrize("n", [20, 30])
def test_built_in(n):
    coarse_grid = mesh.create_rectangle(
        MPI.COMM_WORLD, np.array([[0.0, 0.0], [3.0, 3.0]]), [n, n]
    )

    petsc_vectors = []

    degree = 2

    fe = element("Lagrange", coarse_grid.basix_cell(), degree, shape=(2,))
    W = fem.functionspace(coarse_grid, fe)
    w = fem.Function(W)
    w.vector.array[:] = 1.0

    subdomain_endpoints = [
        ([0.0, 0.0], [1.0, 1.0]),
        ([1.0, 0.0], [2.0, 1.0]),
        ([0.0, 1.0], [1.0, 2.0]),
        ([2.0, 0.0], [3.0, 1.0]),
        ([1.0, 1.0], [2.0, 2.0]),
        ([0.0, 2.0], [1.0, 3.0]),
        ([2.0, 1.0], [3.0, 2.0]),
        ([1.0, 2.0], [2.0, 3.0]),
        ([2.0, 2.0], [3.0, 3.0]),
    ]

    for ci in range(9):
        start, end = subdomain_endpoints[ci]

        subdomain = mesh.create_rectangle(MPI.COMM_WORLD, np.array([start, end]), [n, n])
        fel = element("Lagrange", subdomain.basix_cell(), degree, shape=(2,))
        V = fem.functionspace(subdomain, fel)
        dim = V.dofmap.bs * V.dofmap.index_map.size_global

        # ### interpolation of w for current cell
        u = fem.Function(V, name=f"u_{ci}")
        u.interpolate(w, nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
            u.function_space.mesh,
            u.function_space.element,
            w.function_space.mesh))

        uvec = u.vector
        assert dim == int(uvec.norm(0))

if __name__ == "__main__":
    test_built_in(20)
