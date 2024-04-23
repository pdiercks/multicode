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
from multi.problems import LinElaSubProblem
from multi.materials import LinearElasticMaterial


def test():
    """Test creation of FE space on discretization of boundary of a RectangularSubdomain"""

    # the dof layout of the FE space cannot be guaranteed to be the same
    # for opposing edges (e.g. bottom-top)
    # the reason is most likely the numbering of the entities (gmsh, create_submesh)
    # Therefore, mappings between edge spaces are done using multi.interpolation.interpolate

    num_cells = 10
    gdim = 2

    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_unit_cell_01(
            0.0,
            1.0,
            0.0,
            1.0,
            num_cells=num_cells,
            cell_tags={"matrix": 1, "inclusion": 2},
            facet_tags={"bottom": 1, "left": 2, "right": 3, "top": 4},
            out_file=tf.name,
        )

        rce, ct, ft = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=gdim)
    Ω = RectangularSubdomain(1, rce, ct, ft)
    Ω.create_coarse_grid(1)
    Ω.create_boundary_grids()

    V = fem.functionspace(rce, ("Lagrange", 2, (2,)))
    material = LinearElasticMaterial(gdim, E=20e3, NU=0.3)
    subproblem = LinElaSubProblem(Ω, V, material)

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

    # ### Build mappings
    subproblem.create_edge_space_maps()

    xdofs_b = x_dofs_vectorspace(Vb)
    xdofs_t = x_dofs_vectorspace(Vt)
    ttb = subproblem.edge_space_maps["top_to_bottom"]
    btt = subproblem.edge_space_maps["bottom_to_top"]
    assert np.allclose(xdofs_b[:, 0], xdofs_t[ttb, 0])
    assert np.allclose(xdofs_b[btt, 0], xdofs_t[:, 0])

    xdofs_l = x_dofs_vectorspace(Vl)
    xdofs_r = x_dofs_vectorspace(Vr)
    rtl = subproblem.edge_space_maps["right_to_left"]
    ltr = subproblem.edge_space_maps["left_to_right"]
    assert np.allclose(xdofs_l[:, 1], xdofs_r[rtl, 1])
    assert np.allclose(xdofs_l[ltr, 1], xdofs_r[:, 1])
