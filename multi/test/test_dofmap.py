"""test dofmap"""

import numpy as np
import pygmsh
from multi import DofMap


def build_mesh(NX, NY, LCAR=0.1, order=1):
    geom = pygmsh.built_in.Geometry()
    geom.add_raw_code("Mesh.SecondOrderIncomplete = 1;")
    XO = 0.0
    YO = 0.0
    X = 1.0 * NX
    Y = 1.0 * NY
    square = geom.add_polygon(
        [[XO, YO, 0.0], [X, YO, 0.0], [X, Y, 0.0], [XO, Y, 0.0]], LCAR
    )

    geom.set_transfinite_surface(square.surface, size=[NX + 1, NY + 1])
    geom.add_raw_code("Recombine Surface {%s};" % square.surface.id)
    geom.add_physical([square.surface], label="square")

    mshfile = None
    geofile = None
    mesh = pygmsh.generate_mesh(
        geom,
        dim=2,
        geo_filename=geofile,
        msh_filename=mshfile,
        prune_z_0=True,
        extra_gmsh_arguments=["-order", f"{order}"],
    )
    return mesh


def test_array():
    """test dofmap with type(dofs_per_edge)==np.ndarray"""
    mesh = build_mesh(2, 1, order=2)
    """mesh.points

         0<-------------------->2

    1    3-----10----8-----9----2
    ^    |           |          |
    |    11  cell_0  12  cell_1 7
    v    |           |          |
    0    0-----5-----4-----6----1

    dofs are distributed first for cell 0 in local order of
    vertices and edges.
    Then continue with vertices and edges in local order which
    haven't been assigned any DoFs yet.
    """
    # 6 vertices
    # 7 edges
    n_vertex_dofs = 2
    dofs_per_edge = np.array(
        [
            [5, 12, 10, 11],
            [6, 7, 9, 12],
        ]
    )

    dofmap = DofMap(mesh, 2, 2)
    dofmap.distribute_dofs(n_vertex_dofs, dofs_per_edge, 0)

    N = dofmap.dofs()
    assert N == n_vertex_dofs * 6 + np.sum(dofs_per_edge) - 12
    A = np.zeros((N, N))  # global matrix
    summe = 0
    for ci, cell in enumerate(dofmap.cells):
        n = 4 * n_vertex_dofs + np.sum(dofs_per_edge[ci])
        a = np.ones((n, n))  # local matrix
        summe += n ** 2
        cell_dofs = dofmap.cell_dofs(ci)
        A[np.ix_(cell_dofs, cell_dofs)] += a
    assert np.isclose(np.sum(A), summe)

    x_dofs = dofmap.tabulate_dof_coordinates()
    assert np.allclose(x_dofs[[0, 1]], np.zeros((2, 2)))
    assert np.allclose(x_dofs[list(range(8, 8 + 5))], np.vstack(((0.5, 0.0),) * 5))
    assert np.allclose(x_dofs[list(range(13, 13 + 12))], np.vstack(((1.0, 0.5),) * 12))

    assert np.allclose(
        dofmap.locate_dofs([[0, 0], [0.5, 0]]), np.array([0, 1, 8, 9, 10, 11, 12])
    )
    assert np.allclose(dofmap.locate_dofs([[0, 0], [2, 0]], sub=0), np.array([0, 46]))
    assert np.allclose(
        dofmap.locate_dofs([[2.0, 0.5], [1.0, 1.0]]),
        np.array([56, 57, 58, 59, 60, 61, 62, 4, 5]),
    )
    assert np.allclose(dofmap.locate_cells([[0, 0], [0.5, 0], [0, 0.5]]), [0])
    assert np.allclose(dofmap.locate_cells([[1, 0]]), [0, 1])
    assert np.allclose(dofmap.plane_at(0.0, "x"), np.array([[0, 0], [0, 1], [0, 0.5]]))
    assert np.allclose(
        dofmap.plane_at(0.0, "x", vertices_only=True), np.array([[0, 0], [0, 1]])
    )
    assert np.allclose(dofmap.plane_at(0.0, "x", edges_only=True), np.array([0, 0.5]))

    assert np.allclose(
        dofmap.get_cell_points(
            [
                0,
            ]
        ),
        dofmap.points[dofmap.cells[0]],
    )
    assert np.allclose(
        dofmap.get_cell_points(
            [
                0,
            ],
            gmsh_nodes=[4, 6],
        ),
        np.array([[0.5, 0.0], [0.5, 1.0]]),
    )
    assert np.allclose(
        dofmap.get_cell_points([0, 1], gmsh_nodes=[0, 1, 6]),
        np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, 0.0], [1.5, 1.0]]),
    )


def test_array_uniform():
    """test dofmap with type(dofs_per_edge)==np.ndarray"""
    mesh = build_mesh(2, 1, order=2)
    """mesh.points

    3-----10----8-----9----2
    |           |          |
    11  cell_0  12  cell_1 7
    |           |          |
    0-----5-----4-----6----1
    """
    # 6 vertices
    # 7 edges
    n_vertex_dofs = 2
    n_edge_dofs = 3
    dofs_per_edge = np.ones((2, 4), dtype=int) * n_edge_dofs
    # n_edge_dofs = np.array([
    #     [5, 12, 10, 11],
    #     [6, 7, 9, 12]])

    dofmap = DofMap(mesh, 2, 2)
    dofmap.distribute_dofs(n_vertex_dofs, dofs_per_edge, 0)

    N = dofmap.dofs()
    assert N == n_vertex_dofs * 6 + np.sum(dofs_per_edge) - n_edge_dofs
    A = np.zeros((N, N))
    n = 4 * n_vertex_dofs + 4 * n_edge_dofs
    a = np.ones((n, n))
    for ci, cell in enumerate(dofmap.cells):
        cell_dofs = dofmap.cell_dofs(ci)
        A[np.ix_(cell_dofs, cell_dofs)] += a
    assert np.isclose(np.sum(A), 800)
    x_dofs = dofmap.tabulate_dof_coordinates()
    assert np.allclose(x_dofs[0], np.array([0, 0]))
    assert np.allclose(x_dofs[13], np.array([1.0, 0.5]))
    assert np.allclose(x_dofs[32], np.array([1.5, 1.0]))

    assert np.allclose(
        dofmap.locate_dofs([[0, 0], [0.5, 0]]), np.array([0, 1, 8, 9, 10])
    )
    assert np.allclose(dofmap.locate_dofs([[0, 0], [2, 0]], sub=0), np.array([0, 20]))
    assert np.allclose(dofmap.locate_cells([[0, 0], [0.5, 0], [0, 0.5]]), [0])
    assert np.allclose(dofmap.locate_cells([[1, 0]]), [0, 1])
    assert np.allclose(dofmap.plane_at(0.0, "x"), np.array([[0, 0], [0, 1], [0, 0.5]]))
    assert np.allclose(
        dofmap.plane_at(0.0, "x", vertices_only=True), np.array([[0, 0], [0, 1]])
    )
    assert np.allclose(dofmap.plane_at(0.0, "x", edges_only=True), np.array([0, 0.5]))

    assert np.allclose(
        dofmap.get_cell_points(
            [
                0,
            ]
        ),
        dofmap.points[dofmap.cells[0]],
    )
    assert np.allclose(
        dofmap.get_cell_points(
            [
                0,
            ],
            gmsh_nodes=[4, 6],
        ),
        np.array([[0.5, 0.0], [0.5, 1.0]]),
    )
    assert np.allclose(
        dofmap.get_cell_points([0, 1], gmsh_nodes=[0, 1, 6]),
        np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, 0.0], [1.5, 1.0]]),
    )


def test():
    """test dofmap with type(dofs_per_edge)==int"""
    mesh = build_mesh(2, 1, order=2)
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
    n_edge_dofs = 3

    dofmap = DofMap(mesh, 2, 2)
    dofmap.distribute_dofs(n_vertex_dofs, n_edge_dofs, 0)

    N = dofmap.dofs()
    assert N == n_vertex_dofs * 6 + n_edge_dofs * 7
    A = np.zeros((N, N))
    n = 4 * n_vertex_dofs + 4 * n_edge_dofs
    a = np.ones((n, n))
    for ci, cell in enumerate(dofmap.cells):
        cell_dofs = dofmap.cell_dofs(ci)
        A[np.ix_(cell_dofs, cell_dofs)] += a
    assert np.isclose(np.sum(A), 800)
    x_dofs = dofmap.tabulate_dof_coordinates()
    assert np.allclose(x_dofs[0], np.array([0, 0]))
    assert np.allclose(x_dofs[13], np.array([1.0, 0.5]))
    assert np.allclose(x_dofs[32], np.array([1.5, 1.0]))

    assert np.allclose(
        dofmap.locate_dofs([[0, 0], [0.5, 0]]), np.array([0, 1, 8, 9, 10])
    )
    assert np.allclose(dofmap.locate_dofs([[0, 0], [2, 0]], sub=0), np.array([0, 20]))
    assert np.allclose(dofmap.locate_cells([[0, 0], [0.5, 0], [0, 0.5]]), [0])
    assert np.allclose(dofmap.locate_cells([[1, 0]]), [0, 1])
    assert np.allclose(dofmap.plane_at(0.0, "x"), np.array([[0, 0], [0, 1], [0, 0.5]]))
    assert np.allclose(
        dofmap.plane_at(0.0, "x", vertices_only=True), np.array([[0, 0], [0, 1]])
    )
    assert np.allclose(dofmap.plane_at(0.0, "x", edges_only=True), np.array([0, 0.5]))

    assert np.allclose(
        dofmap.get_cell_points(
            [
                0,
            ]
        ),
        dofmap.points[dofmap.cells[0]],
    )
    assert np.allclose(
        dofmap.get_cell_points(
            [
                0,
            ],
            gmsh_nodes=[4, 6],
        ),
        np.array([[0.5, 0.0], [0.5, 1.0]]),
    )
    assert np.allclose(
        dofmap.get_cell_points([0, 1], gmsh_nodes=[0, 1, 6]),
        np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, 0.0], [1.5, 1.0]]),
    )


if __name__ == "__main__":
    test()
    test_array_uniform()
    test_array()
