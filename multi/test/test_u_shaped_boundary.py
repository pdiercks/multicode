import numpy as np
import dolfin as df
from multi.boundary import u_shaped_boundary_factory

# import matplotlib.pyplot as plt


def test():
    mesh = df.UnitSquareMesh(9, 9)
    V = df.FunctionSpace(mesh, "CG", 2)
    boundary = u_shaped_boundary_factory("x", 1.0)
    bc = df.DirichletBC(V, df.Constant(0.0), boundary)
    dofs = np.array(list(bc.get_boundary_values().keys()))
    x_dofs = V.tabulate_dof_coordinates()
    x = x_dofs[dofs]
    assert np.all(x[:, 0] < 1.0)

    # NOTE dolfin excludes all dofs on cells which are not inside boundary def
    # --> therefore for degree=2 dofs on cell mid-points are excluded

    bc_all = df.DirichletBC(V, df.Constant(0.0), df.DomainBoundary())
    all_boundary_dofs = np.array(list(bc_all.get_boundary_values().keys()))
    diff = np.setdiff1d(all_boundary_dofs, dofs)
    # 9 cells and degree=2 gives 9 * 2 + 1 dofs on the right edge
    # add the 2 mid-point dofs which were also excluded (see note)
    assert np.isclose(diff.size, 9 * 2 + 1 + 2)

    # plt.figure(1)
    # df.plot(mesh)
    # for x, y in x_dofs[dofs]:
    #     plt.plot(x, y, "bo")
    # plt.show()


if __name__ == "__main__":
    test()
