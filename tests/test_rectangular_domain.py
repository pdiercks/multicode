import tempfile
import pytest
from mpi4py import MPI
from dolfinx.io import gmshio
from multi.domain import RectangularDomain
from multi.preprocessing import create_rectangle


def test():
    n = 4
    ftags = {"bottom": 1, "left": 2, "right": 3, "top": 4}
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(0., 1., 0., 1., num_cells=(n, n), facet_tags=ftags, recombine=True, out_file=tf.name)
        domain, _, ft = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    omega = RectangularDomain(domain, cell_tags=None, facet_tags=ft)
    for i in range(1, 5):
        entities = omega.facet_tags.find(i)
        assert entities.size == n

    with pytest.warns():
        omega.create_facet_tags({"bottom": 256, "right": 12})

    assert omega.facet_tags.find(256).size == n
    assert omega.facet_tags.find(12).size == n
    assert omega.facet_tags.find(1).size == 0
    assert omega.facet_tags.find(2).size == 0
    assert omega.facet_tags.find(3).size == 0
    assert omega.facet_tags.find(4).size == 0

    Ω = RectangularDomain(domain)
    with pytest.raises(ValueError):
        Ω.create_facet_tags({"bttom": 2})


if __name__ == "__main__":
    test()

