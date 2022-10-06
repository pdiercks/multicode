import dolfinx
from mpi4py import MPI
from multi.bcs import get_boundary_dofs


def test():
    n = 8
    degree = 2
    dim = 2

    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.quadrilateral
    )
    V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", degree), dim=dim)
    dofs = get_boundary_dofs(V)
    assert dofs.size == n * 4 * degree * dim
    # there are (n+1) * 4 - 4 points
    # and if degree == 2: additional n*4 dofs (edges)
    # thus n * 4 * degree dofs for degree in (1, 2)
    # times number of components (i.e. dim)


if __name__ == "__main__":
    test()
