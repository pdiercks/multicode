from mpi4py import MPI
import dolfinx
from basix.ufl import element
import numpy as np
from multi.boundary import point_at


def test_function_space():
    n = 101
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.quadrilateral
    )
    fe = element("Lagrange", domain.basix_cell(), 2, shape=())
    V = dolfinx.fem.FunctionSpace(domain, fe)

    h = 1.0 / n
    my_point = point_at(np.array([h * 2, h * 5, 0.0]))

    dofs = dolfinx.fem.locate_dofs_geometrical(V, my_point)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(42), dofs, V)
    ndofs = bc._cpp_object.dof_indices()[1]
    assert ndofs == 1
    assert bc.g.value == 42


def test_vector_function_space():
    n = 101
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.quadrilateral
    )
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", 2))

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    nodal_dofs = np.array([], dtype=np.int32)
    for x in points:
        dofs = dolfinx.fem.locate_dofs_geometrical(V, point_at(x))
        bc = dolfinx.fem.dirichletbc(np.array([0, 0], dtype=dolfinx.default_scalar_type), dofs, V)
        nodal_dofs = np.append(nodal_dofs, bc._cpp_object.dof_indices()[0])
    assert nodal_dofs.size == 8


if __name__ == "__main__":
    test_function_space()
    test_vector_function_space()
