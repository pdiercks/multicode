import pytest
import dolfinx
from mpi4py import MPI


@pytest.mark.parametrize("n", [20, 30])
def test_built_in(n):
    coarse_grid = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [[0.0, 0.0], [3.0, 3.0]], [n, n]
    )

    petsc_vectors = []

    degree = 2

    W = dolfinx.fem.VectorFunctionSpace(coarse_grid, ("P", degree))
    w = dolfinx.fem.Function(W)
    w.x.set(1.0)

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

        subdomain = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [start, end], [n, n])
        V = dolfinx.fem.VectorFunctionSpace(subdomain, ("P", degree))
        dim = V.dofmap.bs * V.dofmap.index_map.size_global
        print(f"{dim=}")

        # ### interpolation of w for current cell
        u = dolfinx.fem.Function(V, name=f"u_{ci}")
        u.interpolate(w)

        uvec = u.vector

        petsc_vectors.append(uvec)

    for ci, x in enumerate(petsc_vectors):
        print(f"{ci=},    {x.norm(0)=}")


if __name__ == "__main__":
    test_built_in()
