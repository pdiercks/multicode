import dolfinx
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from multi.boundary import point_at


def test():
    n = 101
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.quadrilateral
    )
    V = dolfinx.fem.FunctionSpace(domain, ("CG", 2))

    h = 1.0 / n
    my_point = point_at([h * 2, h * 5, 0.0])

    dofs = dolfinx.fem.locate_dofs_geometrical(V, my_point)
    bc = dolfinx.fem.dirichletbc(ScalarType(42), dofs, V)
    ndofs = bc.dof_indices()[1]
    assert ndofs == 1
    assert bc.g.value == 42


if __name__ == "__main__":
    test()
