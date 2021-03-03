"""test inner product"""

import ufl
import dolfin as df
from multi import InnerProduct


def test():
    mesh = df.UnitSquareMesh(4, 4)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    prod = InnerProduct(V, "mass", bcs=(), form=None)
    assert isinstance(prod.get_form(), ufl.form.Form)
    assert isinstance(prod.assemble(), df.cpp.la.Matrix)

    prod = InnerProduct(V, None)
    assert prod.get_form() is None
    assert prod.assemble() is None


if __name__ == "__main__":
    test()
