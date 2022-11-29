"""test inner product"""

import ufl
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
from multi.product import InnerProduct


def test():
    domain = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
    names = ("mass", "l2", "h1-semi", "stiffness", "h1")
    for name in names:
        prod = InnerProduct(V, name, bcs=())
        assert isinstance(prod.get_form(), ufl.form.Form)
        assert isinstance(prod.assemble_matrix(), PETSc.Mat)

    A = prod.assemble_matrix()
    assert A.size == (11, 11)
    assert A[0, 0] > 0.0

    prod = InnerProduct(V, "euclidean")
    assert prod.get_form() is None
    assert prod.assemble_matrix() is None

    prod = InnerProduct(V, None)
    assert prod.get_form() is None
    assert prod.assemble_matrix() is None


if __name__ == "__main__":
    test()
