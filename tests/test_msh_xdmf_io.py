import pathlib
import tempfile
import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
from multi.preprocessing import create_rectangle
from numpy import allclose


def test():
    tmp_msh = tempfile.NamedTemporaryFile(suffix=".msh")
    tmp_xdmf = tempfile.NamedTemporaryFile(suffix=".xdmf")

    create_rectangle(
        0.0, 1.0, 0.0, 1.0, num_cells=5, recombine=True, out_file=tmp_msh.name
    )

    domain, _, _ = gmshio.read_from_msh(tmp_msh.name, MPI.COMM_WORLD, gdim=2)

    with dolfinx.io.XDMFFile(domain.comm, tmp_xdmf.name, "w") as xdmf:
        xdmf.write_mesh(domain)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, tmp_xdmf.name, "r") as xdmf:
        other = xdmf.read_mesh()

    # delete h5
    h5 = pathlib.Path(tmp_xdmf.name).with_suffix(".h5")
    h5.unlink()

    tmp_msh.close()
    tmp_xdmf.close()

    assert allclose(domain.geometry.x, other.geometry.x)


if __name__ == "__main__":
    test()
