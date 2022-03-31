import numpy as np
import dolfin as df
from multi.boundary import l_shaped_boundary_factory


def test():
    mesh = df.UnitSquareMesh(9, 9)
    V = df.FunctionSpace(mesh, "CG", 2)
    boundary = l_shaped_boundary_factory(0, [0.0, 1.0], [0.0, 1.0])
    bc = df.DirichletBC(V, df.Constant(0.0), boundary)
    dofs = np.array(list(bc.get_boundary_values().keys()))
    x_dofs = V.tabulate_dof_coordinates()
    x = x_dofs[dofs]
    right = x[x[:, 0] >= 1.0]
    top = x[x[:, 1] >= 1.0]
    assert np.allclose(right[:, 0], np.ones_like(right[:, 0]))
    assert np.allclose(top[:, 1], np.ones_like(top[:, 1]))

    # NOTE dolfin excludes all dofs on cells which are not inside boundary def
    # --> therefore for degree=2 dofs on cell mid-points are excluded

    bc_all = df.DirichletBC(V, df.Constant(0.0), df.DomainBoundary())
    all_boundary_dofs = np.array(list(bc_all.get_boundary_values().keys()))
    diff = np.setdiff1d(all_boundary_dofs, dofs)
    # 9 cells and degree=2 gives 9 * 2 + 1 = 19 dofs on the right edge
    # plus 18 dofs on the top edge
    # on both edges the most outward cells are excluded, thus 4 dofs are excluded
    # thus dofs.size shoulb be 19+18-4=33
    assert np.isclose(diff.size, (9 * 4 + 1) + 2)


if __name__ == "__main__":
    test()
