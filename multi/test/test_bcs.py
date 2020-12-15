"""test MechanicsBCs"""

import dolfin as df
import numpy as np
from multi import MechanicsBCs
from multi import Domain
from fenics_helpers.boundary import plane_at, point_at


def test():
    mesh = df.UnitSquareMesh(6, 6)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    domain = Domain(mesh, 0)
    bc_handler = MechanicsBCs(domain, V)
    left = plane_at(0, "x")
    right = plane_at(1, "x")
    origin = point_at((0, 0))

    bc_handler.add_bc(left, df.Constant(0), sub=0)
    bc_handler.add_bc(origin, df.Constant(0), sub=1, method="pointwise")
    bc_handler.add_force(right, df.Constant((100, 0)))

    bcs = bc_handler.bcs()
    F = df.assemble(bc_handler.boundary_forces())

    assert np.isclose(np.sum(F[:]), 100)
    assert len(bcs) == 2
    assert bc_handler.has_forces

    dofs = []
    vals = []
    for bc in bcs:
        dofs += list(bc.get_boundary_values().keys())
        vals += list(bc.get_boundary_values().values())
    assert np.sum(vals) < 1e-3
    x_dofs = V.tabulate_dof_coordinates()
    assert np.sum(x_dofs[dofs, 0]) < 1e-3


if __name__ == "__main__":
    test()
