"""test inner product"""

import ufl
import dolfin as df
from multi import InnerProduct


def test():
    mesh = df.UnitSquareMesh(4, 4)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    names = ("mass", "l2", "h1-semi", "stiffness", "h1")
    for name in names:
        prod = InnerProduct(V, name, bcs=())
        assert isinstance(prod.get_form(), ufl.form.Form)
        assert isinstance(prod.assemble(), df.cpp.la.Matrix)

    prod = InnerProduct(V, "euclidean")
    assert prod.get_form() is None
    assert prod.assemble() is None

    prod = InnerProduct(V, None)
    assert prod.get_form() is None
    assert prod.assemble() is None


if __name__ == "__main__":
    test()
