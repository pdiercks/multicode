"""test make mapping"""
import dolfin as df
from multi import make_mapping, RectangularDomain
from numpy import allclose


def test():
    domain = RectangularDomain("data/rvedomain.xdmf", edges=True)
    V = df.FunctionSpace(domain.mesh, "CG", 2)
    L = df.FunctionSpace(domain.edges[0], "CG", 2)
    V_to_L = make_mapping(L, V)

    x_dofs = V.tabulate_dof_coordinates()
    bottom = L.tabulate_dof_coordinates()
    assert allclose(bottom, x_dofs[V_to_L])


if __name__ == "__main__":
    test()
