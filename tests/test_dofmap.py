"""test dofmap"""

import tempfile
import pytest

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import gmshio
from basix.ufl import element

from multi.dofmap import DofMap
from multi.domain import StructuredQuadGrid
from multi.preprocessing import create_rectangle_grid


def test():
    """test dofmap with type(dofs_per_edge)==int"""
    n = 8
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(
            0.0, 1.0, 0.0, 1.0, num_cells=(n, n), recombine=True, out_file=tf.name
        )
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    n_vertex_dofs = 2
    n_edge_dofs = 0
    n_face_dofs = 0

    domain.topology.create_connectivity(0, 0)
    domain.topology.create_connectivity(1, 1)

    conn_00 = domain.topology.connectivity(0, 0)
    conn_11 = domain.topology.connectivity(1, 1)
    num_vertices = conn_00.num_nodes
    num_edges = conn_11.num_nodes

    grid = StructuredQuadGrid(domain)
    dofmap = DofMap(grid)

    # dofs are not distributed yet
    with pytest.raises(AttributeError):
        print(f"{dofmap.num_dofs=}")
    with pytest.raises(AttributeError):
        print(f"{dofmap.cell_dofs(0)=}")

    dofmap.distribute_dofs(n_vertex_dofs, n_edge_dofs, n_face_dofs)

    assert dofmap.num_dofs == n_vertex_dofs * num_vertices + n_edge_dofs * num_edges

    ve = element("P", domain.basix_cell(), 1, shape=(2,))
    V = fem.functionspace(domain, ve)

    def get_global_dofs(V, cell_index):
        """get global dofs of V for cell index"""
        bs = V.dofmap.bs
        num_dofs_local = V.dofmap.cell_dofs(0).size
        dofs = []
        cell_dofs = V.dofmap.cell_dofs(cell_index)
        for i in range(num_dofs_local):
            for k in range(bs):
                dofs.append(bs * cell_dofs[i] + k)
        return np.array(dofs)

    # NOTE
    # if n_edge_dofs = n_face_dofs = 0, V.dofmap and multi.DofMap should
    # have the same dof layout
    # V.element.needs_dof_transformations is False --> okay
    u = fem.Function(V)
    u.interpolate(lambda x: (44.7 * x[0], -12.3 * x[1]))
    u_values = u.vector.array

    # FIXME
    # that still does not check whether the dofs
    # have the same coordinates ...?

    for ci in range(dofmap.num_cells):
        mydofs = dofmap.cell_dofs(ci)
        expected = get_global_dofs(V, ci)
        assert np.allclose(mydofs, expected)
        assert np.allclose(u_values[mydofs], u_values[expected])

    # locate entities and get entity dofs
    def bottom(x):
        return np.isclose(x[1], 0.0)

    ents = dofmap.grid.locate_entities(1, bottom)
    assert ents.size == n
    # this should return an empty list
    # since zero dofs per edge are distributed
    dofs = dofmap.entity_dofs(1, ents[0])
    assert not dofs


def test_array_uniform():
    """test dofmap with type(dofs_per_edge)==np.ndarray"""
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(
            0.0, 1.0, 0.0, 1.0, num_cells=(2, 1), recombine=True, out_file=tf.name
        )
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    # 6 vertices
    # 7 edges
    n_vertex_dofs = 2
    n_edge_dofs = 3
    dofs_per_edge = np.ones((2, 4), dtype=int) * n_edge_dofs
    n_face_dofs = 0

    domain.topology.create_connectivity(0, 0)
    domain.topology.create_connectivity(1, 1)

    conn_00 = domain.topology.connectivity(0, 0)
    conn_11 = domain.topology.connectivity(1, 1)
    num_vertices = conn_00.num_nodes
    num_edges = conn_11.num_nodes

    grid = StructuredQuadGrid(domain)
    dofmap = DofMap(grid)
    dofmap.distribute_dofs(n_vertex_dofs, dofs_per_edge, n_face_dofs)

    assert dofmap.num_dofs == n_vertex_dofs * num_vertices + n_edge_dofs * num_edges
    N = dofmap.num_dofs

    num_verts_cell = 4
    num_edges_cell = 4
    A = np.zeros((N, N))
    n = num_verts_cell * n_vertex_dofs + num_edges_cell * n_edge_dofs
    a = np.ones((n, n))
    for ci in range(dofmap.num_cells):
        cell_dofs = dofmap.cell_dofs(ci)
        A[np.ix_(cell_dofs, cell_dofs)] += a
    assert np.isclose(np.sum(A), 800)

    def left(x):
        return np.isclose(x[0], 0.0)

    ents = dofmap.grid.locate_entities(1, left)
    assert ents.size == 1
    dofs = dofmap.entity_dofs(1, ents[0])
    assert len(dofs) == 3
    assert np.allclose(np.array([11, 12, 13]), dofs)

    def bottom(x):
        return np.isclose(x[1], 0.0)

    ents = dofmap.grid.locate_entities(1, bottom)
    assert ents.size == 2
    dofs = dofmap.entity_dofs(1, ents[0])
    assert len(dofs) == 3
    assert np.allclose(np.array([8, 9, 10]), dofs)


def test_array():
    """test dofmap with type(dofs_per_edge)==np.ndarray"""
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(
            0.0, 1.0, 0.0, 1.0, num_cells=(2, 1), recombine=True, out_file=tf.name
        )
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    """
    v2----e3----v3
    |           |
    e1          e2
    |           |
    v0----e0----v1

    """
    # 6 vertices
    # 7 edges
    n_vertex_dofs = 2
    dofs_per_edge = np.array([[5, 12, 10, 11], [11, 10, 7, 12]])
    n_face_dofs = 0

    domain.topology.create_connectivity(0, 0)
    domain.topology.create_connectivity(1, 1)

    conn_00 = domain.topology.connectivity(0, 0)
    num_vertices = conn_00.num_nodes

    grid = StructuredQuadGrid(domain)
    dofmap = DofMap(grid)
    dofmap.distribute_dofs(n_vertex_dofs, dofs_per_edge, n_face_dofs)

    N = dofmap.num_dofs
    # num dofs minus dofs on the middle edge that are not distributed twice
    expected_N = (
        n_vertex_dofs * num_vertices + np.sum(dofs_per_edge) - dofs_per_edge[1][1]
    )
    assert N == expected_N

    num_verts_cell = 4
    A = np.zeros((N, N))  # global matrix
    summe = 0
    for ci in range(dofmap.num_cells):
        n = num_verts_cell * n_vertex_dofs + np.sum(dofs_per_edge[ci])
        a = np.ones((n, n))  # local matrix
        summe += n**2
        cell_dofs = dofmap.cell_dofs(ci)
        A[np.ix_(cell_dofs, cell_dofs)] += a
    assert np.isclose(np.sum(A), summe)

    def origin(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

    def top(x):
        return np.isclose(x[1], 1.0)

    vertex = dofmap.grid.locate_entities(0, origin)
    assert len(vertex) == 1
    dofs = dofmap.entity_dofs(0, vertex[0])
    assert np.allclose(np.array([0, 1]), np.array(dofs))

    edges = dofmap.grid.locate_entities_boundary(1, top)
    dofs = []
    for edge in edges:
        dofs += dofmap.entity_dofs(1, edge)
    assert len(dofs) == np.sum(dofs_per_edge[:, 3])
    distributed_until_top_0 = 8 + np.sum(dofs_per_edge[0, :3])
    distributed_until_top_1 = (
        8
        + np.sum(dofs_per_edge[0, :])
        + 4
        + np.sum(dofs_per_edge[1, :3])
        - dofs_per_edge[1, 1]
    )
    assert np.allclose(
        np.hstack(
            (
                np.arange(dofs_per_edge[0, 3]) + distributed_until_top_0,
                np.arange(dofs_per_edge[1, 3]) + distributed_until_top_1,
            )
        ),
        dofs,
    )


if __name__ == "__main__":
    test()
    test_array_uniform()
    test_array()
