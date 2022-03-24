"""test boundary conditions"""

import dolfin as df
import numpy as np
from multi import BoundaryConditions
from multi import Domain
from fenics_helpers.boundary import plane_at, point_at


def test_t1_c1():
    """
    topological dimension: 1
    (vector) field dimension: 1
    """
    mesh = df.UnitIntervalMesh(10)
    V = df.FunctionSpace(mesh, "CG", 1)
    domain = Domain(mesh)
    bc_handler = BoundaryConditions(domain, V)
    origin = point_at((0,))
    end = point_at((1.0,))
    bc_handler.add_dirichlet(origin, df.Constant(0.0), method="pointwise")
    bc_handler.set_zero(end, "0", method="pointwise")

    bcs = bc_handler.bcs()
    dofs = []
    vals = []
    for bc in bcs:
        dofs += list(bc.get_boundary_values().keys())
        vals += list(bc.get_boundary_values().values())
    assert len(bcs) == 2
    assert len(dofs) == 2
    assert np.sum(vals) < 1e-9


def test_t2_c2():
    """
    topological dimension: 2
    (vector) field dimension: 2
    """
    mesh = df.UnitSquareMesh(6, 6)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    domain = Domain(mesh, 0)
    bc_handler = BoundaryConditions(domain, V)
    left = plane_at(0, "x")
    right = plane_at(1, "x")
    origin = point_at((0, 0))

    bc_handler.add_dirichlet(left, df.Constant(0), sub=0)
    bc_handler.add_dirichlet(origin, df.Constant(0), sub=1, method="pointwise")
    bc_handler.add_neumann(right, df.Constant((100, 0)))

    bcs = bc_handler.bcs()
    F = df.assemble(bc_handler.neumann_bcs())

    assert np.isclose(np.sum(F[:]), 100)
    assert len(bcs) == 2
    assert bc_handler.has_neumann()

    dofs = []
    vals = []
    for bc in bcs:
        dofs += list(bc.get_boundary_values().keys())
        vals += list(bc.get_boundary_values().values())
    assert np.sum(vals) < 1e-3
    x_dofs = V.tabulate_dof_coordinates()
    assert np.sum(x_dofs[dofs, 0]) < 1e-3


if __name__ == "__main__":
    test_t1_c1()
    # TODO
    # test_t1_c2()
    # test_t2_c1()
    test_t2_c2()
