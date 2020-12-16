"""test ResultFile"""
import dolfin as df
from numpy import allclose
from multi.io import ResultFile


def test():
    mesh = df.UnitSquareMesh(8, 8)
    V = df.FunctionSpace(mesh, "CG", 1)

    alpha = 3
    beta = 1.2
    expr = df.Expression(
        "1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t",
        degree=2,
        alpha=alpha,
        beta=beta,
        t=0,
    )

    f = df.Function(V)
    out = ResultFile("data/resultfile.xdmf")
    out.add_function(f, name="f")

    values = []
    for j in range(5):
        t = float(j / 5)
        expr.t = t
        g = df.interpolate(expr, V)
        values.append(g.vector().get_local())
        f.assign(g)
        out.write_checkpoint("f", t)
    out.close()

    # read back in
    d = df.Function(V)
    infile = ResultFile("data/resultfile.xdmf")
    other = []
    assert d.vector().max() == 0.0
    for j in range(5):
        infile.read_checkpoint(d, "f", j)
        other.append(d.vector().get_local())
    infile.close()
    assert all([allclose(v, o) for (v, o) in zip(values, other)])


if __name__ == "__main__":
    test()
