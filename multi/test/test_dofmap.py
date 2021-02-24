"""test dofmap"""

import numpy as np
import pygmsh
from multi import DofMap


def build_mesh(NX, NY, LCAR=0.1, order=1):
    geom = pygmsh.built_in.Geometry()
    geom.add_raw_code("Mesh.SecondOrderIncomplete = 1;")
    XO = 0.0
    YO = 0.0
    X = 1.0
    Y = 1.0
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


def test():
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
    assert np.allclose(x_dofs[13], np.array([0.5, 0.5]))
    assert np.allclose(x_dofs[32], np.array([0.75, 1.0]))

    assert np.allclose(
        dofmap.locate_dofs([[0, 0], [0.25, 0]]), np.array([0, 1, 8, 9, 10])
    )
    assert np.allclose(dofmap.locate_dofs([[0, 0], [1, 0]], sub=0), np.array([0, 20]))
    assert np.allclose(dofmap.locate_cells([[0, 0], [0.25, 0], [0, 0.5]]), [0])
    assert np.allclose(dofmap.locate_cells([[0.5, 0]]), [0, 1])
    assert np.allclose(dofmap.plane_at(0.0, "x"), np.array([[0, 0], [0, 1], [0, 0.5]]))


if __name__ == "__main__":
    test()
