"""
plot condition number of gramian against number of modes for basis

Usage:
    gramian.py [options] RVE DEG BASES...

Arguments:
    RVE       The XDMF file for the RVE domain (incl. ext).
    DEG       Degree of FE space.
    BASES     The reduced bases (incl. .npy extension).

Options:
    -h, --help               Show this message and exit.
    --product=PROD           An inner product (see `discretize_rve`) [default: energy_0].
    -o FILE, --output=FILE   Write PDF to path.
"""
import sys
from pathlib import Path
from docopt import docopt

import yaml
from plotstuff import PlottingContext
import numpy as np

import dolfin as df
from fenicsphelpers.subdomains import Subdomain
from fenicsphelpers.linear_elasticity import LinearElasticityProblem

from pymor.bindings.fenics import FenicsMatrixOperator, FenicsVectorSpace

figures = Path(__file__).parent
root = figures.absolute().parent
computations = root / "computations"


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["RVE"] = Path(args["RVE"])
    args["DEG"] = int(args["DEG"])
    args["BASES"] = [Path(d) for d in args["BASES"]]
    assert all([d.exists() for d in args["BASES"]])
    return args


def main(args):
    args = parse_arguments(args)
    # BAM colors
    with open(figures / "bamcolors_hex.yml") as instream:
        bamcd = yaml.safe_load(instream)

    source, products = discretize_rve(args)
    bases = []
    labels = []
    for npy in args["BASES"]:
        labels.append(npy.stem.split("_")[0])
        rb = np.load(npy)
        RB = source.from_numpy(rb)
        bases.append(RB)

    gramians = []
    product = products[args["--product"]]
    for basis in bases:
        G = basis.gramian(product)
        gramians.append(G)

    kappas = {}
    for (i, (basis, G)) in enumerate(zip(bases, gramians)):
        kappas[i] = []
        for N in range(1, len(basis)):
            kappas[i].append(np.linalg.cond(G[:N, :N]))

    colors = [bamcd["BAMred1"], bamcd["BAMblue2"], bamcd["BAMgreen1"]]
    markers = ["x", "+", "<"]
    plot_argv = [__file__, args["--output"]] if args["--output"] else [__file__]
    with PlottingContext(plot_argv, "pdiercks_article") as fig:
        ax = fig.subplots()
        for key, value in kappas.items():
            ax.semilogy(
                value, color=colors[key], marker=markers[key], label=labels[key]
            )
        ax.set_xlabel(r"Number of modes $N$")
        ax.set_ylabel("Condition number of the gramian")
        ax.legend()
        ax.grid()


def discretize_rve(args):
    """discretize the rve and wrap as pyMOR model"""
    rve_xdmf = args["RVE"]
    rve_domain = Subdomain(rve_xdmf, 0, subdomains=True)
    with open(computations / "material.yml", "r") as infile:
        try:
            material = yaml.safe_load(infile)
        except yaml.YAMLError as exc:
            print(exc)

    E = material["Material parameters"]["E"]["value"]
    NU = material["Material parameters"]["NU"]["value"]

    V = df.VectorFunctionSpace(rve_domain.mesh, "CG", args["DEG"])
    problem = LinearElasticityProblem(rve_domain, V, E=E, NU=NU)
    a = problem.get_lhs()
    A = df.assemble(a)

    # boundary data g
    #  q = df.Function(V)
    g = df.Function(V)

    def boundary(x, on_boundary):
        return on_boundary

    null = np.zeros(V.dim())
    S = FenicsVectorSpace(V)

    g.vector().set_local(null)
    A_bc = A.copy()
    bc = df.DirichletBC(V, g, boundary)
    bc.apply(A_bc)
    # NOTE empirical and hierarchical basis are orthonormal wrt to
    # A_bc using bc.apply since this is used when solving for psi.

    # ### products
    energy_mat = A.copy()
    energy_0_mat = A_bc.copy()
    l2_mat = problem.get_product(name="l2", bcs=False)
    l2_0_mat = l2_mat.copy()
    h1_mat = problem.get_product(name="h1", bcs=False)
    h1_0_mat = h1_mat.copy()

    bc.apply(l2_0_mat)
    bc.apply(h1_0_mat)

    products = {
        "energy": FenicsMatrixOperator(energy_mat, V, V, name="energy"),
        "energy_0": FenicsMatrixOperator(energy_0_mat, V, V, name="energy_0"),
        "l2": FenicsMatrixOperator(l2_mat, V, V, name="l2"),
        "l2_0": FenicsMatrixOperator(l2_0_mat, V, V, name="l2_0"),
        "h1": FenicsMatrixOperator(h1_mat, V, V, name="h1"),
        "h1_0": FenicsMatrixOperator(h1_0_mat, V, V, name="h1_0"),
    }
    return S, products


if __name__ == "__main__":
    main(sys.argv[1:])
