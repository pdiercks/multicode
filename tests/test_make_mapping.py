"""Test make_mapping."""

import pytest
import tempfile
import pathlib

from mpi4py import MPI
import dolfinx as df
from multi.interpolation import make_mapping
from multi.preprocessing import create_rectangle
from multi.io import read_mesh
from multi.boundary import within_range
from multi.misc import x_dofs_vectorspace
import numpy as np


@pytest.mark.parametrize("value_shape", [(), (2,)])
def test(value_shape):

    xmin = 12.4
    xmax = 99.78
    ymin = -1.45
    ymax = 45.93
    num_cells = [100, 70]
    gdim = 2

    with tempfile.NamedTemporaryFile(suffix='.msh') as tf:
        create_rectangle(xmin, xmax, ymin, ymax, num_cells=num_cells, recombine=True, out_file=tf.name, options={"Mesh.ElementOrder": 2})
        omega, _, _ = read_mesh(pathlib.Path(tf.name), MPI.COMM_WORLD, kwargs={"gdim": gdim})

    target_marker = within_range([xmin + (xmax-xmin)/3, ymin + (ymax-ymin)/3, 0.], [xmin + (xmax-xmin)*2/3, ymin + (ymax-ymin)*2/3, 0.])
    cells_subdomain = df.mesh.locate_entities(omega, omega.topology.dim, target_marker)
    submesh, _, _, _ = df.mesh.create_submesh(omega, omega.topology.dim, cells_subdomain)
    print(cells_subdomain.size)

    V = df.fem.functionspace(omega, ("P", 2, value_shape))
    W = df.fem.functionspace(submesh, V.ufl_element())
    dofmap = make_mapping(W, V)
    xdofs_v = x_dofs_vectorspace(V)
    xdofs_w = x_dofs_vectorspace(W)
    err = xdofs_w - xdofs_v[dofmap]
    assert np.linalg.norm(err) < 1e-9

    with tempfile.NamedTemporaryFile(suffix=".xdmf") as tf:
        with df.io.utils.XDMFFile(submesh.comm, tf.name, "w") as xdmf:
            xdmf.write_mesh(submesh)
        # reload, and reconstruct FE space
        omega_in = read_mesh(pathlib.Path(tf.name), MPI.COMM_WORLD)[0]

    X = df.fem.functionspace(omega_in, V.ufl_element())
    xdofs_x = x_dofs_vectorspace(X)
    err = xdofs_w - xdofs_x
    assert np.linalg.norm(err) < 1e-9


if __name__ == "__main__":
    test(())
    test((2,))
