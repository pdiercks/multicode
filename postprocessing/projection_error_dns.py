"""
compute projection error for given basis wrt dns

Usage:
    projection_error_dns.py [options] FINE DNS COARSE RCE DEG MAT BASIS

Arguments:
    FINE         The fine scale grid of the global problem.
    DNS          The direct numerical simulation (xdmf).
    COARSE       The coarse scale grid.
    RCE          The rce grid.
    DEG          Degree of FE space.
    MAT          Material (yml) data.
    BASIS        The reduced basis (npz).

Options:
    -h, --help               Show this message and exit.
    -l LEVEL, --log=LEVEL    Set the log level [default: 30].
    --output=TXT             Write projection error to TXT.
    --plot-errors            Plot the projection error against modes.
    --cells=NPY              Restrict DNS snapshot set to given cells. 
"""

import sys
from pathlib import Path
from docopt import docopt

import yaml
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from multi import Domain, ResultFile, DofMap, LinearElasticityProblem
from multi.misc import read_basis

from pymor.bindings.fenics import FenicsVectorSpace, FenicsMatrixOperator
from pymor.core.logger import getLogger


def parse_args(args):
    args = docopt(__doc__, args)
    args["FINE"] = Path(args["FINE"])
    args["DNS"] = Path(args["DNS"])
    args["COARSE"] = Path(args["COARSE"])
    args["RCE"] = Path(args["RCE"])
    args["DEG"] = int(args["DEG"])
    args["MAT"] = Path(args["MAT"])
    args["BASIS"] = Path(args["BASIS"])
    args["--output"] = Path(args["--output"]) if args["--output"] is not None else None
    args["--log"] = int(args["--log"])
    args["--cells"] = Path(args["--cells"]) if args["--cells"] is not None else None
    return args


def read_dns(args, logger):
    domain = Domain(args["FINE"], id_=0, subdomains=True)
    V = df.VectorFunctionSpace(domain.mesh, "CG", args["DEG"])
    d = df.Function(V)

    logger.info(f"Reading DNS from file {args['DNS']} ...")
    xdmf = ResultFile(args["DNS"].absolute())
    xdmf.read_checkpoint(d, "u-dns", 0)
    xdmf.close()
    return d


def get_snapshots(args, logger, dns, source):
    dofmap = DofMap(args["COARSE"], 2, 2)
    active_cells = np.arange(len(dofmap.cells))
    if args["--cells"]:
        cells = np.load(args["--cells"])
        assert set(cells).issubset(active_cells)
        active_cells = np.array(list(set(active_cells).difference(cells)))

    snaps = source.empty(reserce=len(dofmap.cells))
    logger.info("Gathering snapshot data ...")
    for cell_index, cell in enumerate(dofmap.cells[active_cells]):
        offset = np.around(dofmap.points[cell][0], decimals=5)
        omega = Domain(
            args["RCE"], id_=cell_index, subdomains=True, translate=df.Point(offset)
        )
        V = df.VectorFunctionSpace(omega.mesh, "CG", args["DEG"])
        Idns = df.interpolate(dns, V)
        snaps.append(source.make_array([Idns.vector()]))

    return snaps


# this function is part of the pymor tutorial https://docs.pymor.org/2020.2.0/tutorial_basis_generation.html
def compute_proj_errors(basis, V, product):
    G = basis.gramian(product=product)
    R = basis.inner(V, product=product)
    errors = []
    for N in range(len(basis) + 1):
        if N > 0:
            v = np.linalg.solve(G[:N, :N], R[:N, :])
        else:
            # N = 0, such that err is the norm of V
            v = np.zeros((0, len(V)))
        V_proj = basis[:N].lincomb(v.T)
        err = (V - V_proj).norm(product=product)
        errors.append(np.max(err))
    return errors


def main(args):
    args = parse_args(args)
    logger = getLogger("main")
    logger.setLevel(args["--log"])
    dns = read_dns(args, logger)

    # rce problem to define vector space and product
    rce_domain = Domain(args["RCE"], 99, subdomains=True)
    V = df.VectorFunctionSpace(rce_domain.mesh, "CG", args["DEG"])
    with open(args["MAT"], "r") as f:
        mat = yaml.safe_load(f)
    E = mat["Material parameters"]["E"]["value"]
    NU = mat["Material parameters"]["NU"]["value"]
    plane_stress = mat["Constraints"]["plane_stress"]
    problem = LinearElasticityProblem(
        rce_domain, V, E=E, NU=NU, plane_stress=plane_stress
    )

    energy = problem.get_product(name="energy", bcs=False)
    h1_mat = problem.get_product(name="h1", bcs=False)
    products = [
        None,
        FenicsMatrixOperator(h1_mat, V, V, name="h1"),
        FenicsMatrixOperator(energy, V, V, name="energy"),
    ]

    S = FenicsVectorSpace(V)
    snapshots = get_snapshots(args, logger, dns, S)
    logger.info(f"Reading basis from file {args['BASIS']} ...")
    rb, nm = read_basis(args["BASIS"])
    basis = S.from_numpy(rb)

    with logger.block("Computing projection errors ..."):
        errors = []
        names = []
        for prod in products:
            proj_errs = compute_proj_errors(basis, snapshots, prod)
            try:
                name = prod.name
            except AttributeError:
                name = "euclidean"
            names.append(name)
            errors.append(proj_errs)

    if args["--plot-errors"]:
        plt.figure(1)
        plt.title("projection error")
        for err, name in zip(errors, names):
            nmodes = np.arange(len(err))
            plt.semilogy(nmodes, err, "--*", label=f"{name}")
        plt.grid()
        plt.legend()
        plt.show()

    if args["--output"]:
        with open(args["--output"], "w") as out:
            np.savetxt(out, np.vstack(errors).T, delimiter=",", header=", ".join(names))


if __name__ == "__main__":
    main(sys.argv[1:])
