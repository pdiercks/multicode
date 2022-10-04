"""test dofmap"""

import dolfinx
import numpy as np
from mpi4py import MPI
from multi.dofmap import DofMap

# from helpers import build_mesh


# def test_array():
#     """test dofmap with type(dofs_per_edge)==np.ndarray"""
#     mesh = build_mesh(2, 1, order=2)
#     """mesh.points

#          0<-------------------->2

#     1    3-----10----8-----9----2
#     ^    |           |          |
#     |    11  cell_0  12  cell_1 7
#     v    |           |          |
#     0    0-----5-----4-----6----1

#     dofs are distributed first for cell 0 in local order of
#     vertices and edges.
#     Then continue with vertices and edges in local order which
#     haven't been assigned any DoFs yet.
#     """
#     # 6 vertices
#     # 7 edges
#     n_vertex_dofs = 2
#     dofs_per_edge = np.array(
#         [
#             [5, 12, 10, 11],
#             [6, 7, 9, 12],
#         ]
#     )

#     dofmap = DofMap(mesh, 2, 2)
#     dofmap.distribute_dofs(n_vertex_dofs, dofs_per_edge, 0)

#     N = dofmap.dofs()
#     assert N == n_vertex_dofs * 6 + np.sum(dofs_per_edge) - 12
#     A = np.zeros((N, N))  # global matrix
#     summe = 0
#     for ci, cell in enumerate(dofmap.cells):
#         n = 4 * n_vertex_dofs + np.sum(dofs_per_edge[ci])
#         a = np.ones((n, n))  # local matrix
#         summe += n**2
#         cell_dofs = dofmap.cell_dofs(ci)
#         A[np.ix_(cell_dofs, cell_dofs)] += a
#     assert np.isclose(np.sum(A), summe)

#     x_dofs = dofmap.tabulate_dof_coordinates()
#     assert np.allclose(x_dofs[[0, 1]], np.zeros((2, 2)))
#     assert np.allclose(x_dofs[list(range(8, 8 + 5))], np.vstack(((0.5, 0.0),) * 5))
#     assert np.allclose(x_dofs[list(range(13, 13 + 12))], np.vstack(((1.0, 0.5),) * 12))

#     assert np.allclose(
#         dofmap.locate_dofs([[0, 0], [0.5, 0]]), np.array([0, 1, 8, 9, 10, 11, 12])
#     )
#     xxx = dofmap.within_range([0.0, 0.0], [0.5, 0.0])
#     c1 = np.allclose(dofmap.locate_dofs(xxx), np.array([0, 1, 8, 9, 10, 11, 12]))
#     xxx = dofmap.within_range([0.0, 0.0], [0.5, 0.0], edges_only=True)
#     c2 = np.allclose(dofmap.locate_dofs(xxx, sub=0), np.array([8, 10, 12]))
#     xxx = dofmap.within_range([0.0, 0.0], [0.5, 0.0], vertices_only=True)
#     c3 = np.allclose(
#         dofmap.locate_dofs(xxx, sub=1),
#         np.array(
#             [
#                 1,
#             ]
#         ),
#     )
#     assert all([c1, c2, c3])
#     assert np.allclose(dofmap.locate_dofs([[0, 0], [2, 0]], sub=0), np.array([0, 46]))
#     assert np.allclose(
#         dofmap.locate_dofs([[2.0, 0.5], [1.0, 1.0]]),
#         np.array([56, 57, 58, 59, 60, 61, 62, 4, 5]),
#     )
#     assert np.allclose(dofmap.locate_cells([[0, 0], [0.5, 0], [0, 0.5]]), [0])
#     assert np.allclose(dofmap.locate_cells([[1, 0]]), [0, 1])
#     assert np.allclose(dofmap.plane_at(0.0, "x"), np.array([[0, 0], [0, 1], [0, 0.5]]))
#     assert np.allclose(
#         dofmap.plane_at(0.0, "x", vertices_only=True), np.array([[0, 0], [0, 1]])
#     )
#     assert np.allclose(dofmap.plane_at(0.0, "x", edges_only=True), np.array([0, 0.5]))

#     assert np.allclose(
#         dofmap.get_cell_points(
#             [
#                 0,
#             ]
#         ),
#         dofmap.points[dofmap.cells[0]],
#     )
#     assert np.allclose(
#         dofmap.get_cell_points(
#             [
#                 0,
#             ],
#             gmsh_nodes=[4, 6],
#         ),
#         np.array([[0.5, 0.0], [0.5, 1.0]]),
#     )
#     assert np.allclose(
#         dofmap.get_cell_points([0, 1], gmsh_nodes=[0, 1, 6]),
#         np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, 0.0], [1.5, 1.0]]),
#     )


# def test_array_uniform():
#     """test dofmap with type(dofs_per_edge)==np.ndarray"""
#     mesh = build_mesh(2, 1, order=2)
#     """mesh.points

#     3-----10----8-----9----2
#     |           |          |
#     11  cell_0  12  cell_1 7
#     |           |          |
#     0-----5-----4-----6----1
#     """
#     # 6 vertices
#     # 7 edges
#     n_vertex_dofs = 2
#     n_edge_dofs = 3
#     dofs_per_edge = np.ones((2, 4), dtype=int) * n_edge_dofs
#     # n_edge_dofs = np.array([
#     #     [5, 12, 10, 11],
#     #     [6, 7, 9, 12]])

#     dofmap = DofMap(mesh, 2, 2)
#     dofmap.distribute_dofs(n_vertex_dofs, dofs_per_edge, 0)

#     N = dofmap.dofs()
#     assert N == n_vertex_dofs * 6 + np.sum(dofs_per_edge) - n_edge_dofs
#     A = np.zeros((N, N))
#     n = 4 * n_vertex_dofs + 4 * n_edge_dofs
#     a = np.ones((n, n))
#     for ci, cell in enumerate(dofmap.cells):
#         cell_dofs = dofmap.cell_dofs(ci)
#         A[np.ix_(cell_dofs, cell_dofs)] += a
#     assert np.isclose(np.sum(A), 800)
#     x_dofs = dofmap.tabulate_dof_coordinates()
#     assert np.allclose(x_dofs[0], np.array([0, 0]))
#     assert np.allclose(x_dofs[13], np.array([1.0, 0.5]))
#     assert np.allclose(x_dofs[32], np.array([1.5, 1.0]))

#     assert np.allclose(
#         dofmap.locate_dofs([[0, 0], [0.5, 0]]), np.array([0, 1, 8, 9, 10])
#     )
#     assert np.allclose(dofmap.locate_dofs([[0, 0], [2, 0]], sub=0), np.array([0, 20]))
#     assert np.allclose(dofmap.locate_cells([[0, 0], [0.5, 0], [0, 0.5]]), [0])
#     assert np.allclose(dofmap.locate_cells([[1, 0]]), [0, 1])
#     assert np.allclose(dofmap.plane_at(0.0, "x"), np.array([[0, 0], [0, 1], [0, 0.5]]))
#     assert np.allclose(
#         dofmap.plane_at(0.0, "x", vertices_only=True), np.array([[0, 0], [0, 1]])
#     )
#     assert np.allclose(dofmap.plane_at(0.0, "x", edges_only=True), np.array([0, 0.5]))

#     assert np.allclose(
#         dofmap.get_cell_points(
#             [
#                 0,
#             ]
#         ),
#         dofmap.points[dofmap.cells[0]],
#     )
#     assert np.allclose(
#         dofmap.get_cell_points(
#             [
#                 0,
#             ],
#             gmsh_nodes=[4, 6],
#         ),
#         np.array([[0.5, 0.0], [0.5, 1.0]]),
#     )
#     assert np.allclose(
#         dofmap.get_cell_points([0, 1], gmsh_nodes=[0, 1, 6]),
#         np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, 0.0], [1.5, 1.0]]),
#     )


def test():
    """test dofmap with type(dofs_per_edge)==int"""
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 2, 1, dolfinx.mesh.CellType.quadrilateral
    )
    """mesh.points

    3-----10----8-----9----2
    |           |          |
    11          12         7
    |           |          |
    0-----5-----4-----6----1
    """
    # 6 vertices
    # 7 edges
    n_vertex_dofs = 2
    n_edge_dofs = 0

    domain.topology.create_connectivity(0, 0)
    domain.topology.create_connectivity(1, 1)

    conn_00 = domain.topology.connectivity(0, 0)
    conn_11 = domain.topology.connectivity(1, 1)
    num_vertices = conn_00.num_nodes
    num_edges = conn_11.num_nodes

    dofmap = DofMap(domain)
    dofmap.distribute_dofs(n_vertex_dofs, n_edge_dofs, 0)

    assert dofmap.num_dofs() == n_vertex_dofs * num_vertices + n_edge_dofs * num_edges

    V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
    # TODO dofmap.cell_dofs()

    # NOTE
    # I want to achieve the same DofMap as V.dofmap in case n_edge_dofs = n_face_dofs = 0
    # and n_vertex_dofs = 2
    # However, V.dofmap.cell_dofs(i) always returns an array of length 4 instead of 8,
    # because internally V.dofmap.bs is somehow (somewhere) used

    # TODO
    # I could test this by assembling a matrix or vector

    from IPython import embed

    embed()

    # N = dofmap.dofs()
    # assert N == n_vertex_dofs * 6 + n_edge_dofs * 7
    # A = np.zeros((N, N))
    # n = 4 * n_vertex_dofs + 4 * n_edge_dofs
    # a = np.ones((n, n))
    # for ci, cell in enumerate(dofmap.cells):
    #     cell_dofs = dofmap.cell_dofs(ci)
    #     A[np.ix_(cell_dofs, cell_dofs)] += a
    # assert np.isclose(np.sum(A), 800)
    # x_dofs = dofmap.tabulate_dof_coordinates()
    # assert np.allclose(x_dofs[0], np.array([0, 0]))
    # assert np.allclose(x_dofs[13], np.array([1.0, 0.5]))
    # assert np.allclose(x_dofs[32], np.array([1.5, 1.0]))

    # assert np.allclose(
    #     dofmap.locate_dofs([[0, 0], [0.5, 0]]), np.array([0, 1, 8, 9, 10])
    # )
    # assert np.allclose(dofmap.locate_dofs([[0, 0], [2, 0]], sub=0), np.array([0, 20]))
    # assert np.allclose(dofmap.locate_cells([[0, 0], [0.5, 0], [0, 0.5]]), [0])
    # assert np.allclose(dofmap.locate_cells([[1, 0]]), [0, 1])
    # assert np.allclose(dofmap.plane_at(0.0, "x"), np.array([[0, 0], [0, 1], [0, 0.5]]))
    # assert np.allclose(
    #     dofmap.plane_at(0.0, "x", vertices_only=True), np.array([[0, 0], [0, 1]])
    # )
    # assert np.allclose(dofmap.plane_at(0.0, "x", edges_only=True), np.array([0, 0.5]))

    # assert np.allclose(
    #     dofmap.get_cell_points(
    #         [
    #             0,
    #         ]
    #     ),
    #     dofmap.points[dofmap.cells[0]],
    # )
    # assert np.allclose(
    #     dofmap.get_cell_points(
    #         [
    #             0,
    #         ],
    #         gmsh_nodes=[4, 6],
    #     ),
    #     np.array([[0.5, 0.0], [0.5, 1.0]]),
    # )
    # assert np.allclose(
    #     dofmap.get_cell_points([0, 1], gmsh_nodes=[0, 1, 6]),
    #     np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, 0.0], [1.5, 1.0]]),
    # )


if __name__ == "__main__":
    test()
    # test_array_uniform()
    # test_array()
