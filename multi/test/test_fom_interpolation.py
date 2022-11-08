import dolfinx
from mpi4py import MPI

""" 31.10.2022

- issue does not occurr with built-in meshes
- it does not occur if I print u.vetor.norm() inside the loop
- it does not occur if I use u.x instead of u.vector

"""


def test():
    global_grid = "/mnt/paper/work/block/grids/global_fine_grid.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, global_grid, "r") as x:
        mesh = x.read_mesh(name="Grid")

    petsc_vectors = []

    degree = 2

    W = dolfinx.fem.VectorFunctionSpace(mesh, ("P", degree))
    w = dolfinx.fem.Function(W)
    w.x.set(1.0)

    n = 9
    cells = list(range(n))

    for ci in cells:

        subdomain_file = f"/mnt/paper/work/block/grids/offline/subdomain_{ci:03}.xdmf"
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, subdomain_file, "r") as xdmf:
            subdomain = xdmf.read_mesh(name="Grid")

        print(subdomain_file)

        V = dolfinx.fem.VectorFunctionSpace(subdomain, ("P", degree))
        # dim = V.dofmap.bs * V.dofmap.index_map.size_global

        # ### interpolation of FOM solution for current cell
        u = dolfinx.fem.Function(V)
        u.x.set(0.0)
        u.interpolate(w)

        uvec = u.vector
        # u.x.scatter_forward()
        # print(u.x.norm())

        # print(f"{ci=},    {uvec.norm(0)=}")
        petsc_vectors.append(uvec)

    for ci, x in enumerate(petsc_vectors):
        print(f"{ci=},    {x.norm(0)=}")


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
    test_built_in(20)
    # test_built_in(20)
