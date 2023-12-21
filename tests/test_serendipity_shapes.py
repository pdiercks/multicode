"""test serendipity shape functions"""

import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
import tempfile
import numpy as np
from multi.domain import Domain
from multi.preprocessing import create_rectangle
from multi.shapes import NumpyQuad


def test():
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(
            0.0, 2.0, 0.0, 2.0, num_cells=(20, 20), recombine=True, out_file=tf.name
        )
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    omega = Domain(domain)
    omega.translate([-1, -1, 0])
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2))
    quad8 = NumpyQuad(
        np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    )
    shapes = quad8.interpolate(V)

    def serendipity(xi, eta):
        """return shape function at 8th node"""
        return (1 - xi) * (1 - eta**2) / 2

    assert np.isclose(np.sum(shapes), len(V.tabulate_dof_coordinates()))
    x_dofs = V.tabulate_dof_coordinates()
    x = x_dofs[:, 0]
    y = x_dofs[:, 1]

    u = serendipity(x, y)
    e = u - shapes[7]
    assert np.linalg.norm(e) < 1e-14


if __name__ == "__main__":
    test()
