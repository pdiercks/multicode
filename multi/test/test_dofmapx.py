"""test dofmap"""

import dolfinx
import numpy as np
from mpi4py import MPI
from multi.dofmap import DofMap


def test():
    """test dofmap with type(dofs_per_edge)==int"""
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, dolfinx.mesh.CellType.quadrilateral
    )
    n_vertex_dofs = 2
    n_edge_dofs = 0
    n_face_dofs = 0

    domain.topology.create_connectivity(0, 0)
    domain.topology.create_connectivity(1, 1)

    conn_00 = domain.topology.connectivity(0, 0)
    conn_11 = domain.topology.connectivity(1, 1)
    num_vertices = conn_00.num_nodes
    num_edges = conn_11.num_nodes

    dofmap = DofMap(domain)
    dofmap.distribute_dofs(n_vertex_dofs, n_edge_dofs, n_face_dofs)

    assert dofmap.num_dofs() == n_vertex_dofs * num_vertices + n_edge_dofs * num_edges

    V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))

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
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: (44.7 * x[0], -12.3 * x[1]))
    u_values = u.vector.array

    for ci in range(dofmap.num_cells):
        mydofs = dofmap.cell_dofs(ci)
        expected = get_global_dofs(V, ci)
        assert np.allclose(mydofs, expected)
        assert np.allclose(u_values[mydofs], u_values[expected])


def test_array_uniform():
    """test dofmap with type(dofs_per_edge)==np.ndarray"""
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 2, 1, dolfinx.mesh.CellType.quadrilateral
    )
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

    dofmap = DofMap(domain)
    dofmap.distribute_dofs(n_vertex_dofs, dofs_per_edge, n_face_dofs)

    assert dofmap.num_dofs() == n_vertex_dofs * num_vertices + n_edge_dofs * num_edges
    N = dofmap.num_dofs()

    num_verts_cell = 4
    num_edges_cell = 4
    A = np.zeros((N, N))
    n = num_verts_cell * n_vertex_dofs + num_edges_cell * n_edge_dofs
    a = np.ones((n, n))
    for ci in range(dofmap.num_cells):
        cell_dofs = dofmap.cell_dofs(ci)
        A[np.ix_(cell_dofs, cell_dofs)] += a
    assert np.isclose(np.sum(A), 800)


def test_array():
    """test dofmap with type(dofs_per_edge)==np.ndarray"""
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 2, 1, dolfinx.mesh.CellType.quadrilateral
    )
    """local cell topology of the domain

    v1----e2----v3
    |           |
    e0          e3
    |           |
    v0----e1----v2

    NOTE that this is different from the layout of the reference cell
    (basix.topology(quadrilateral) & basix.geometry(quadrilateral)
    """
    # 6 vertices
    # 7 edges
    n_vertex_dofs = 2
    dofs_per_edge = np.array([[5, 12, 10, 11], [11, 10, 7, 12]])
    n_face_dofs = 0

    domain.topology.create_connectivity(0, 0)
    domain.topology.create_connectivity(1, 1)

    conn_00 = domain.topology.connectivity(0, 0)
    conn_11 = domain.topology.connectivity(1, 1)
    num_vertices = conn_00.num_nodes
    num_edges = conn_11.num_nodes

    dofmap = DofMap(domain)
    dofmap.distribute_dofs(n_vertex_dofs, dofs_per_edge, n_face_dofs)

    N = dofmap.num_dofs()
    # num dofs minus dofs on the middle edge that are not distributed twice 
    expected_N = n_vertex_dofs * num_vertices + np.sum(dofs_per_edge) - dofs_per_edge[1][0]
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

    dofs = dofmap.entity_dofs(0, 0)
    assert np.allclose(np.array([0, 1]), np.array(dofs))
    dofs = dofmap.entity_dofs(1, 3)
    assert len(dofs) == 11
    assert np.allclose(np.arange(11) + 35, np.array(dofs))


if __name__ == "__main__":
    test()
    test_array_uniform()
    test_array()
