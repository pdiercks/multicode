"""test domain"""

import dolfin as df
import numpy as np
from multi import Domain, RectangularDomain


def get_unit_square_mesh():
    mesh = df.UnitSquareMesh(8, 8)
    subdomains = df.MeshFunction("size_t", mesh, 2, value=0)

    class Left(df.SubDomain):
        def inside(self, x, on_boundary):
            return x[0] <= 0.5

    left = Left()
    left.mark(subdomains, 1)

    return mesh, subdomains


def get_unit_interval_mesh():
    mesh = df.UnitIntervalMesh(10)
    return mesh


def test_1d():
    mesh = get_unit_interval_mesh()
    domain = Domain(mesh)
    xmax = domain.xmax
    ymax = domain.ymax
    zmin = domain.zmin
    assert xmax == 1.0
    assert ymax == 0.0
    assert zmin == 0.0
    domain.translate(df.Point([1.5]))
    assert domain.xmax == 2.5
    assert domain.ymax == 0.0


def test_2d():
    mesh, subs = get_unit_square_mesh()
    domain = Domain(mesh, _id=0, subdomains=subs)
    assert np.isclose(np.sum(domain.subdomains.array()), 64)

    other = Domain(mesh, _id=1, subdomains=None)
    assert other.subdomains is None
    other.translate(df.Point((2.1, 0.4)))
    assert np.isclose(other.ymax, 1.4)
    assert np.isclose(other.xmax, 3.1)

    another = RectangularDomain(
        "data/rvedomain.xdmf", _id=2, subdomains=True, edges=True
    )
    assert len(another.edges) == 4
    assert all([isinstance(e, df.cpp.mesh.Mesh) for e in another.edges])
    assert np.sum(another.subdomains.array()) > 1
    # subdomain numbering is assumed to start with 1 (pygmsh default)
    Ω_i = np.amin(another.subdomains.array())
    assert Ω_i > 0 and Ω_i < 2
    assert np.isclose(another.xmin, 0.0)
    assert np.isclose(another.ymin, 0.0)
    assert np.isclose(another.xmax, another.ymax)


if __name__ == "__main__":
    test_1d()
    test_2d()
