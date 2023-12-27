from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem
from basix.ufl import element
from multi.bcs import get_boundary_dofs
from multi.boundary import within_range


def test():
    n = 3
    degree = 2
    ncomp = 2

    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, mesh.CellType.quadrilateral
    )
    fe = element("Lagrange", domain.basix_cell(), degree, shape=(ncomp,))
    V = fem.functionspace(domain, fe)

    dd = get_boundary_dofs(V)
    ddm = get_boundary_dofs(V, within_range([0., 0.], [1., 1.]))
    assert np.allclose(dd, ddm)

    xdofs = V.tabulate_dof_coordinates()
    xx = xdofs[dd]

    # number of vertices (n+1) * 4 - 4 = n*4
    # if degree == 2: n*4 additional edge midpoints
    # ndofs = n * 4 * ncomp (degree 1)
    # ndofs = n * 8 * ncomp (degree 2)
    # V.tabulate_dof_coordinates() does not take dofmap.bs into account
    # --> xx.size = n * 4 * ncomp (for degree in (1, 2))
    assert len(xx) == n * 4 * ncomp

if __name__ == "__main__":
    test()
