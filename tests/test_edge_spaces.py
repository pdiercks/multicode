from mpi4py import MPI
import tempfile
import dolfinx
import numpy as np
from dolfinx.io import gmshio
from basix.ufl import element

from multi.domain import RectangularSubdomain
from multi.preprocessing import create_unit_cell_01
from multi.interpolation import make_mapping
from multi.misc import x_dofs_vectorspace


def test():
    """Test creation of FE space on discretization of boundary of a RectangularSubdomain"""
    # if multi.domain.RectangularDomain.create_edges uses
    # `dolfinx.mesh.create_submesh` this cannot be guaranteed
    # unfortunately ...
    # Better to still use `create_submesh` and always use appropriate dof mappings, instead
    # of assuming same DOF layout in the code?
    # The assumption of a particular dof layout seems unsafe.

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

    be = element("Lagrange", bottom.basix_cell(), 2, shape=(2,), gdim=bottom.geometry.dim)
    te = element("Lagrange", top.basix_cell(), 2, shape=(2,), gdim=top.geometry.dim)
    le = element("Lagrange", left.basix_cell(), 2, shape=(2,), gdim=left.geometry.dim)
    re = element("Lagrange", right.basix_cell(), 2, shape=(2,), gdim=right.geometry.dim)

    Vb = dolfinx.fem.functionspace(bottom, be)
    Vt = dolfinx.fem.functionspace(top, te)
    Vl = dolfinx.fem.functionspace(left, le)
    Vr = dolfinx.fem.functionspace(right, re)

    xdofs_b = x_dofs_vectorspace(Vb)
    xdofs_t = x_dofs_vectorspace(Vt)
    mapping = make_mapping(Vb, Vt)
    assert np.allclose(xdofs_b[:, 0], xdofs_t[mapping, 0])

    xdofs_l = x_dofs_vectorspace(Vl)
    xdofs_r = x_dofs_vectorspace(Vr)
    mapping = make_mapping(Vl, Vr)
    assert np.allclose(xdofs_l[:, 1], xdofs_r[mapping, 1])


if __name__ == "__main__":
    test()
