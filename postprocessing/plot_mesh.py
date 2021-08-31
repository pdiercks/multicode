"""
plot the given mesh

Usage:
    plot_mesh.py [options] XDMF

Arguments:
    XDMF         The mesh data to be plotted.

Options:
    -h, --help         Show this message and exit.
    --subdomains       Plot subdomains as well.
    --colorbar         Use a colorbar for different subdomains.
    --colormap=CMAP    Choose the colormap [default: viridis].
    --axis-off         Do not plot axes.
    --pdf=FILE         Write result to PDF.
"""

import sys
from docopt import docopt
from pathlib import Path

from dolfin import Mesh, MeshValueCollection, XDMFFile, MeshFunction, plot
from multi.plotting_context import PlottingContext
from matplotlib.colors import ListedColormap
from numpy import load

POSTPROCESSING = Path(__file__).parent


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["XDMF"] = Path(args["XDMF"])
    args["--pdf"] = Path(args["--pdf"]) if args["--pdf"] else None
    args["--colormap"] = str(args["--colormap"])
    supported_cmaps = ("viridis", "bam-RdBu", "RdYlBu")
    if args["--colormap"] not in supported_cmaps:
        print(
            f"Colormap not in supported colormaps {supported_cmaps}. Using defaut value 'viridis'."
        )
        args["--colormap"] = "viridis"
    return args


def main(args):
    args = parse_arguments(args)
    mesh = Mesh()
    mvc = MeshValueCollection("size_t", mesh, dim=None)
    with XDMFFile(args["XDMF"].as_posix()) as f:
        f.read(mesh)
        if args["--subdomains"]:
            f.read(mvc, "gmsh:physical")

    if args["--colormap"] == "bam-RdBu":
        bam_RdBu = load(POSTPROCESSING / "bam-RdBu.npy")
        cmap = ListedColormap(bam_RdBu, name="bam-RdBu")
    elif args["--colormap"] == "RdYlBu":
        cmap = "RdYlBu"
    else:
        cmap = "viridis"

    plot_argv = [__file__, args["--pdf"]] if args["--pdf"] else [__file__]
    with PlottingContext(plot_argv, "pdiercks_multi") as fig:
        if args["--subdomains"]:
            subdomains = MeshFunction("size_t", mesh, mvc)
            ps = plot(subdomains, cmap=cmap)
            if args["--colorbar"]:
                fig.colorbar(ps)
        if args["--axis-off"]:
            import matplotlib.pyplot as plt

            plt.axis("off")
        else:
            ax = fig.axes[0]
            ax.set_xlabel(r"$x$ in $\mathrm{mm}$")
            ax.set_ylabel(r"$y$ in $\mathrm{mm}$")
        plot(mesh)


if __name__ == "__main__":
    main(sys.argv[1:])
