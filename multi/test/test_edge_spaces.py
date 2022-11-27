import tempfile
import dolfinx
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI

from multi.domain import RectangularDomain
from multi.preprocessing import create_rce_grid_01


def test():
    """edge spaces (bottom & top and left & right) should
    have same dof layout"""
    # if multi.domain.RectangularDomain.create_edges uses
    # `dolfinx.mesh.create_submesh` this cannot be guaranteed
    # unfortunately ...

    num_cells = 10
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rce_grid_01(0., 1., 0., 1., num_cells=num_cells, out_file=tf.name)

        rce, ct, ft = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    Ω = RectangularDomain(rce, ct, ft)
    Ω.create_edge_grids(num_cells)

    bottom = Ω.fine_edge_grid["bottom"]
    top = Ω.fine_edge_grid["top"]

    Vb = dolfinx.fem.VectorFunctionSpace(bottom, ("P", 2))
    Vt = dolfinx.fem.VectorFunctionSpace(top, ("P", 2))

    xdofs_b = Vb.tabulate_dof_coordinates()
    xdofs_t = Vt.tabulate_dof_coordinates()
    assert np.allclose(xdofs_b[:, 0], xdofs_t[:, 0])


if __name__ == "__main__":
    test()
