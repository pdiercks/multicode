import dolfinx
from mpi4py import MPI


def test():
    coarse_grid = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 2, [0.0, 10.0])
    V = dolfinx.fem.FunctionSpace(coarse_grid, ("CG", 1))
    u = dolfinx.fem.Function(V)

    fine_grid = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 10, [0.0, 10.0])
    W = dolfinx.fem.FunctionSpace(fine_grid, ("CG", 1))

    u.interpolate(lambda x: x[0] / 10.0)
    from IPython import embed

    embed()


if __name__ == "__main__":
    test()
