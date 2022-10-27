import pytest
import dolfinx
import numpy as np
from mpi4py import MPI
from multi.boundary import plane_at
from multi.bcs import compute_dirichlet_online
from multi.domain import StructuredQuadGrid
from multi.dofmap import DofMap


# cases: inhom, inhom sub, const, const sub, zero, zero sub
def setup():
    n = 4
    coarse_grid = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.quadrilateral)

    grid = StructuredQuadGrid(coarse_grid)
    dofmap = DofMap(grid)

    W = dolfinx.fem.VectorFunctionSpace(coarse_grid, ("P", 2))
    g = dolfinx.fem.Function(W)
    return g, dofmap


def test_g_const():
    g, dofmap = setup()
    # ### g=const.
    bottom = plane_at(0.0, "y")
    float_value = 43.2
    g.x.set(float_value)
    bc = {"value": g, "boundary": bottom, "method": "geometrical"}
    dirichlet = {"inhomogeneous": [bc], "homogeneous": []}

    # if g=const. --> there should be no dof distributed for the edge
    dofmap.distribute_dofs(2, 0)

    r = compute_dirichlet_online(dofmap, dirichlet)
    values = np.array(list(r.values()))
    assert np.isclose(np.sum(values), 10 * float_value)


def test_g_const_sub():
    g, dofmap = setup()
    # ### g_x = const., g_y free
    bottom = plane_at(0.0, "y")
    float_value = 43.2
    g.x.set(0.0)
    g.x.array[::2] = np.full(g.x.array[::2].shape, float_value, dtype=np.float64)
    bc = {"value": g, "boundary": bottom, "method": "geometrical", "sub": 0}
    dirichlet = {"inhomogeneous": [bc], "homogeneous": []}

    dofmap.distribute_dofs(2, 3)

    r = compute_dirichlet_online(dofmap, dirichlet)
    values = np.array(list(r.values()))
    assert np.isclose(np.sum(values), 5 * float_value)


def test_g_hom_assertion_error():
    g, dofmap = setup()
    bottom = plane_at(0.0, "y")
    g.x.set(0.0)
    bc = {"value": g, "boundary": bottom, "method": "geometrical"}
    dirichlet = {"inhomogeneous": [], "homogeneous": [bc]}
    # if g is zero, there should not be any fine scale modes
    # even if there might be any by mistake --> set dofs to zero
    dofmap.distribute_dofs(2, 2)  # should raise AssertionError
    with pytest.raises(AssertionError):
        compute_dirichlet_online(dofmap, dirichlet)


def test_g_hom():
    g, dofmap = setup()
    bottom = plane_at(0.0, "y")
    g.x.set(0.0)
    bc = {"value": g, "boundary": bottom, "method": "geometrical"}
    dirichlet = {"inhomogeneous": [], "homogeneous": [bc]}
    dofmap.distribute_dofs(2, 0)
    r = compute_dirichlet_online(dofmap, dirichlet)
    values = np.array(list(r.values()))
    assert np.isclose(np.sum(values), 0.0)


def test_g_hom_sub():
    g, dofmap = setup()
    bottom = plane_at(0.0, "y")
    g.x.set(13.0)
    g.x.array[1::2] = np.full(g.x.array[1::2].shape, 0.0, dtype=np.float64)
    bc = {"value": g, "boundary": bottom, "method": "geometrical", "sub": 1}
    dirichlet = {"inhomogeneous": [], "homogeneous": [bc]}
    # here edge modes should be present and not prescribed
    # free x-componet in this case
    dofmap.distribute_dofs(2, 2)
    r = compute_dirichlet_online(dofmap, dirichlet)
    values = np.array(list(r.values()))
    assert np.isclose(np.sum(values), 0.0)


def test_g_inhom():
    g, dofmap = setup()
    bottom = plane_at(0.0, "y")
    g.interpolate(lambda x: (x[0] - x[0] * x[0], x[1]))
    bc = {"value": g, "boundary": bottom, "method": "geometrical"}
    dirichlet = {"inhomogeneous": [bc], "homogeneous": []}
    dofmap.distribute_dofs(2, 1)
    r = compute_dirichlet_online(dofmap, dirichlet)
    values = np.array(list(r.values()))

    def f(x):
        return x - x ** 2
    summe = 0.
    for x in np.linspace(0, 1, num=5, endpoint=True):
        summe += f(x)  # coarse dof values
    n_edges = 4
    summe += 1. * n_edges  # fine dof values
    assert np.isclose(np.sum(values), summe)


def test_g_inhom_assertion_error():
    g, dofmap = setup()
    bottom = plane_at(0.0, "y")
    g.interpolate(lambda x: (x[0] - x[0] * x[0], x[1]))
    bc = {"value": g, "boundary": bottom, "method": "geometrical"}
    dirichlet = {"inhomogeneous": [bc], "homogeneous": []}
    dofmap.distribute_dofs(2, 4)  # should give error
    with pytest.raises(AssertionError):
        compute_dirichlet_online(dofmap, dirichlet)


def test_g_inhom_sub():
    g, dofmap = setup()
    bottom = plane_at(0.0, "y")
    g.interpolate(lambda x: (x[0] - x[0] * x[0], x[1]))
    bc = {"value": g, "boundary": bottom, "method": "geometrical", "sub": 0}
    dirichlet = {"inhomogeneous": [bc], "homogeneous": []}
    dofmap.distribute_dofs(2, 1)
    r = compute_dirichlet_online(dofmap, dirichlet)
    values = np.array(list(r.values()))

    def f(x):
        return x - x ** 2
    summe = 0.
    for x in np.linspace(0, 1, num=5, endpoint=True):
        summe += f(x)  # coarse dof values
    n_edges = 4
    summe += 1. * n_edges  # fine dof values
    assert np.isclose(np.sum(values), summe)


if __name__ == "__main__":
    test_g_const()
    test_g_const_sub()
    test_g_hom()
    test_g_hom_sub()
    test_g_hom_assertion_error()
    test_g_inhom_assertion_error()
    test_g_inhom()
    test_g_inhom_sub()
