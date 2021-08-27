"""
plot condition number of gramian against number of modes for basis

Usage:
    gramian.py [options] RVE DEG BASES...
    gramian.py [options] RVE DEG BASES... [-l LABEL]...

Arguments:
    RVE       The XDMF file for the RVE domain (incl. ext).
    DEG       Degree of FE space.
    BASES     The reduced bases (incl. .npy extension).

Options:
    -h, --help               Show this message and exit.
    --product=PROD           An inner product (see `discretize_rve`) [default: energy_0].
    --material=MAT           Material parameters in case energy product is used.
    -o FILE, --output=FILE   Write PDF to path.
    -l, --label=LABEL        Add a label for each data set.
"""
import sys
from pathlib import Path
from docopt import docopt

import yaml
from multi.plotting_context import PlottingContext
import numpy as np

import dolfin as df
from multi import Domain, LinearElasticityProblem
from multi.misc import read_basis

from pymor.bindings.fenics import FenicsMatrixOperator, FenicsVectorSpace

POSTPROCESSING = Path(__file__).parent


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["RVE"] = Path(args["RVE"])
    args["DEG"] = int(args["DEG"])
    args["BASES"] = [Path(d) for d in args["BASES"]]
    assert all([d.exists() for d in args["BASES"]])
    if args["--product"] in ("energy", "energy_0"):
        if args["--material"] is None:
            raise FileNotFoundError(
                "You need to define material parameters for {} product.".format(
                    args["--product"]
                )
            )
    if args["--label"]:
        args["--label"] = [str(lbl) for lbl in args["--label"]]
        assert len(args["--label"]) == len(args["BASES"])
        args["legend"] = True
    else:
        args["--label"] = [
            None,
        ] * len(args["DATA"])
        args["legend"] = False
    return args


def main(args):
    args = parse_arguments(args)

    # BAM colors
    with open(POSTPROCESSING / "bamcolors_hex.yml", "r") as instream:
        bamcd = yaml.safe_load(instream)

    source, products = discretize_rve(args)
    bases = []
    labels = []
    for npz in args["BASES"]:
        rb = read_basis(npz)
        RB = source.from_numpy(rb)
        bases.append(RB)

    # TODO multiple products for one basis

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

    cc = ["blue", "green", "red"]
    keys = []
    for i in range(1, 4):
        for c in cc:
            s = "BAM" + c + f"{i}"
            keys.append(s)

    if len(args["BASES"]) > len(keys):
        raise NotImplementedError

    plot_argv = [__file__, args["--output"]] if args["--output"] else [__file__]
    with PlottingContext(plot_argv, "pdiercks_article") as fig:
        ax = fig.subplots()
        for i, value in kappas.items():
            ax.semilogy(
                value,
                color=bamcd[keys[i]]["c"],
                marker=bamcd[keys[i]]["m"],
                label=args["--label"][i],
            )
        ax.set_xlabel(r"Number of modes $N$")
        ax.set_ylabel("Condition number of the gramian")
        if args["legend"]:
            ax.legend()
        ax.grid()


def discretize_rve(args):
    """discretize the rve and wrap as pyMOR model"""
    rve_xdmf = args["RVE"]
    if args["--material"]:
        rve_domain = Domain(rve_xdmf, 0, subdomains=True)
        with open(args["--material"], "r") as infile:
            try:
                material = yaml.safe_load(infile)
            except yaml.YAMLError as exc:
                print(exc)

        E = material["Material parameters"]["E"]["value"]
        NU = material["Material parameters"]["NU"]["value"]
    else:
        rve_domain = Domain(rve_xdmf, 0, subdomains=False)
        E = 210e3
        NU = 0.3

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
