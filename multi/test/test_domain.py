"""test domain"""

import dolfin as df
import numpy as np
from multi import Domain


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
    domain = Domain(mesh, id_=0, subdomains=subs, edges=False, translate=None)
    assert np.isclose(np.sum(domain.subdomains.array()), 64)
    assert np.isclose(domain.xmin, 0.0)

    other = Domain(
        mesh, id_=1, subdomains=None, edges=False, translate=df.Point((2.1, 0.4))
    )
    assert other.subdomains is None
    assert np.isclose(other.ymax, 1.4)
    assert np.isclose(other.xmax, 3.1)

    another = Domain(
        "data/rvedomain.xdmf", id_=2, subdomains=True, edges=True, translate=None
    )
    assert len(another.edges) == 4
    assert all([isinstance(e, df.cpp.mesh.Mesh) for e in another.edges])
    assert np.sum(another.subdomains.array()) > 1
    # subdomain numbering is assumed to start with 1 (pygmsh default)
    Ω_i = np.amin(another.subdomains.array())
    assert Ω_i > 0 and Ω_i < 2


if __name__ == "__main__":
    test()
