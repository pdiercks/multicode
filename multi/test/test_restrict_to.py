import numpy as np
import dolfin as df
from multi import Domain
from multi.misc import restrict_to, make_mapping


class TargetSubdomain(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= 0.5 + 1e-3 and x[1] <= 0.5 + 1e-3


def test():
    mesh = df.UnitSquareMesh(6, 6)
    V = df.FunctionSpace(mesh, "CG", 2)

    omega_in = TargetSubdomain()
    submesh = df.SubMesh(mesh, omega_in)
    subdomain = Domain(submesh)

    S = df.FunctionSpace(submesh, V.ufl_element())
    V_to_S = make_mapping(S, V)

    funcs = []
    for i in range(10):
        expr = df.Expression(
            "x[0] + x[1] + A * x[0] * x[1]", A=i, degree=V.ufl_element().degree()
        )
        f = df.interpolate(expr, V)
        funcs.append(f)
    r = restrict_to(subdomain, funcs)

    for i in range(10):
        value = np.allclose(funcs[i].vector()[V_to_S], r[i].vector()[:])
        assert value

    r = restrict_to(subdomain, f)
    value = np.allclose(f.vector()[V_to_S], r.vector()[:])
    assert value


if __name__ == "__main__":
    test()
