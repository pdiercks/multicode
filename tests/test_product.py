"""test inner product"""

from mpi4py import MPI
import ufl
import dolfinx
import pytest
from basix.ufl import element
from petsc4py import PETSc
from multi.product import InnerProduct


def test():
    domain = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    fe = element("P", domain.basix_cell(), 1, shape=())
    V = dolfinx.fem.functionspace(domain, fe)
    names = ("mass", "l2", "h1-semi", "stiffness", "h1")
    for name in names:
        prod = InnerProduct(V, name, bcs=())
        assert isinstance(prod.get_form(), ufl.Form)
        assert isinstance(prod.assemble_matrix(), PETSc.Mat)

    prod = InnerProduct(V, "h1") 

    A = prod.assemble_matrix()
    if A is not None:
        assert A.size == (11, 11)
        assert A[0, 0] > 0.0

    prod = InnerProduct(V, "euclidean")
    assert prod.get_form() is None
    assert prod.assemble_matrix() is None

    prod = InnerProduct(V, None)
    assert prod.get_form() is None
    assert prod.assemble_matrix() is None

    # raises_key_error = InnerProduct(V, "euliden")
    with pytest.raises(KeyError):
        InnerProduct(V, "euliden")


if __name__ == "__main__":
    test()
