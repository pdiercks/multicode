"""
surface plots of exemplary bilinear, quadratic and bubble shape functions

Usage:
    plot_basis.py [options] RCE DEG BASIS MODES...

Arguments:
    RCE       The RCE grid (incl. ext).
    DEG       Degree of the FE space.
    BASIS     Filepath incl. extension.
    MODES     A list of integers specifying modes to plot (Note
              that the x-component is plotted by default for all modes).

Options:
    -h, --help               Show this message.
    -o FILE, --output=FILE   Target PDF filepath.
    --component=COMP         Component to plot [default: 0].
"""

import sys
from docopt import docopt
from pathlib import Path

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from multi.plotting_context import PlottingContext
from multi.misc import read_basis

from dolfin import XDMFFile, Mesh, FunctionSpace

POSTPROCESSING = Path(__file__).parent


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["RCE"] = Path(args["RCE"])
    args["DEG"] = int(args["DEG"])
    args["BASIS"] = Path(args["BASIS"])
    args["MODES"] = [int(m) for m in args["MODES"]]
    args["--component"] = int(args["--component"])
    args["--output"] = Path(args["--output"]) if args["--output"] else None
    return args


def main(args):
    args = parse_arguments(args)

    mesh = Mesh()
    with XDMFFile(args["RCE"].as_posix()) as f:
        f.read(mesh)

    V = FunctionSpace(mesh, "CG", args["DEG"])
    x_dofs = V.tabulate_dof_coordinates()
    x = x_dofs[:, 0]
    y = x_dofs[:, 1]

    # load the full basis
    basis = read_basis(args["BASIS"])[0]

    if args["--output"]:
        target_base = args["--output"].stem
        targets = [
            args["--output"].parent / (target_base + f"_{m}.pdf") for m in args["MODES"]
        ]
    sub = np.s_[args["--component"] :: 2]

    # define bam colormap
    bam_RdBu = np.load(POSTPROCESSING / "bam-RdBu.npy")
    bam_cmap = ListedColormap(bam_RdBu, name="bam-RdBu")

    # FIXME if output is None opening multiple figures does not work
    for k, mode in enumerate(args["MODES"]):
        plot_argv = [__file__, targets[k]] if args["--output"] else [__file__]
        with PlottingContext(plot_argv, "pdiercks_multi") as fig:
            z = basis[mode, sub]
            ax = plt.axes(projection="3d")
            ax.plot_trisurf(x, y, z, cmap=bam_cmap, edgecolor="none")
            ax.set_xlabel("x")
            ax.set_ylabel("y")


if __name__ == "__main__":
    main(sys.argv[1:])
