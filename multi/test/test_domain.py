"""test domain"""

import dolfin as df
import numpy as np
from multi import Domain, RectangularDomain


def get_dolfin_mesh():
    mesh = df.UnitSquareMesh(8, 8)
    subdomains = df.MeshFunction("size_t", mesh, 2, value=0)

    class Left(df.SubDomain):
        def inside(self, x, on_boundary):
            return x[0] <= 0.5

    left = Left()
    left.mark(subdomains, 1)

    return mesh, subdomains


def test():
    mesh, subs = get_dolfin_mesh()
    domain = Domain(mesh, _id=0, subdomains=subs)
    assert np.isclose(np.sum(domain.subdomains.array()), 64)

    other = Domain(mesh, _id=1, subdomains=None)
    assert other.subdomains is None
    other.translate(df.Point((2.1, 0.4)))
    assert np.isclose(np.amax(other.mesh.coordinates()[:, 1]), 1.4)
    assert np.isclose(np.amax(other.mesh.coordinates()[:, 0]), 3.1)

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
    test()
