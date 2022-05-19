"""
plot given modes of the edge basis functions

Usage:
    plot_edge_basis.py [options] RCE DEG CHI MODES...

Arguments:
    RCE          The rce grid used.
    DEG          The degree of FE space.
    CHI          TODO
    MODES        TODO

Options:
    -h, --help     Show this message and exit.
    --set=SET      Bottom-Top (0) or Right-Left (1) set. [default: 0]
    --output=PDF   Save to filepath PDF.
"""

import sys
from pathlib import Path
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
from dolfin import FunctionSpace
from multi import Domain


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["RCE"] = Path(args["RCE"])
    args["DEG"] = int(args["DEG"])
    args["CHI"] = Path(args["CHI"])
    args["MODES"] = [int(m) for m in args["MODES"]]
    args["--set"] = str(args["--set"])
    args["--output"] = Path(args["--output"]) if args["--output"] is not None else None
    return args


def main(args):
    args = parse_arguments(args)
    domain = Domain(args["RCE"], 0, subdomains=True, edges=True)
    s = args["--set"]
    edge = domain.edges[int(s)]
    V = FunctionSpace(edge, "CG", args["DEG"])
    x_dofs = V.tabulate_dof_coordinates()[:, int(s)]
    o = np.argsort(x_dofs)
    basis = np.load(args["CHI"])[s]

    if args["--output"] is None:
        # plot without PlottingContext
        fig, axs = plt.subplots(1, 2)
        axs = axs.ravel()
        for i in range(2):
            for m in args["MODES"]:
                axs[i].plot(x_dofs[o], basis[m][i::2][o], "-x")

        plt.tight_layout()
        plt.show()
    else:
        raise NotImplementedError
        from plotstuff import PlottingContext

        plot_argv = [__file__, args["--output"]]
        with PlottingContext(plot_argv, "pdiercks_multi") as fig:
            ax = plt.axes()


if __name__ == "__main__":
    main(sys.argv[1:])
