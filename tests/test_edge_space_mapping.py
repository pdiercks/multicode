"""find new way of mapping from top to bottom without using interpolation"""


from mpi4py import MPI
import tempfile
import numpy as np
from dolfinx.io import gmshio
from dolfinx import fem
from basix.ufl import element

from multi.domain import RectangularSubdomain
from multi.preprocessing import create_unit_cell_01
from multi.misc import x_dofs_vectorspace
from multi.interpolation import interpolate


def test():
    """Test creation of FE space on discretization of boundary of a RectangularSubdomain"""

    # the dof layout of the FE space cannot be guaranteed to be the same
    # for opposing edges (e.g. bottom-top)
    # the reason is most likely the numbering of the entities (gmsh, create_submesh)
    # Therefore, mappings between edge spaces are done using multi.interpolation.interpolate

    num_cells = 10
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_unit_cell_01(0., 1., 0., 1., num_cells=num_cells, cell_tags={"matrix": 1, "inclusion": 2}, facet_tags={"bottom": 1, "left": 2, "right": 3, "top": 4}, out_file=tf.name)

        rce, ct, ft = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    Ω = RectangularSubdomain(1, rce, ct, ft)
    Ω.create_coarse_grid(1)
    Ω.create_boundary_grids()

    bottom = Ω.fine_edge_grid["bottom"]
    top = Ω.fine_edge_grid["top"]
    left = Ω.fine_edge_grid["left"]
    right = Ω.fine_edge_grid["right"]

    be = element("Lagrange", bottom.basix_cell(), 2, shape=(2,))
    te = element("Lagrange", top.basix_cell(), 2, shape=(2,))
    le = element("Lagrange", left.basix_cell(), 2, shape=(2,))
    re = element("Lagrange", right.basix_cell(), 2, shape=(2,))

    Vb = fem.functionspace(bottom, be)
    Vt = fem.functionspace(top, te)
    Vl = fem.functionspace(left, le)
    Vr = fem.functionspace(right, re)

    # ### Map from top to bottom
    f = fem.Function(Vt)
    x_bottom = Vb.tabulate_dof_coordinates()
    x_top = Vt.tabulate_dof_coordinates()
    shift = x_top - x_bottom
    shift[:, 0] *= 0
    f.x.array[:] = np.arange(f.x.array.size, dtype=np.int32)
    values = interpolate(f, x_bottom + shift)
    map = (values.flatten() + 0.5).astype(np.int32)

    xdofs_b = x_dofs_vectorspace(Vb)
    xdofs_t = x_dofs_vectorspace(Vt)
    assert np.allclose(xdofs_b[:, 0], xdofs_t[map, 0])


    # ### Map from right to left
    g = fem.Function(Vr)
    x_left = Vl.tabulate_dof_coordinates()
    x_right = Vr.tabulate_dof_coordinates()
    shift = x_right - x_left
    shift[:, 1] *= 0
    g.x.array[:] = np.arange(g.x.array.size, dtype=np.int32)
    values = interpolate(g, x_left + shift)
    map = (values.flatten() + 0.5).astype(np.int32)

    xdofs_l = x_dofs_vectorspace(Vl)
    xdofs_r = x_dofs_vectorspace(Vr)
    assert np.allclose(xdofs_l[:, 1], xdofs_r[map, 1])
