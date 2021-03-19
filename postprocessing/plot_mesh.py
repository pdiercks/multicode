"""
plot the given mesh

Usage:
    plot_mesh.py [options] XDMF

Arguments:
    XDMF         The mesh data to be plotted.

Options:
    -h, --help     Show this message and exit.
    --subdomains   Plot subdomains as well.
    --colorbar     Use a colorbar for different subdomains.
    --pdf=FILE     Write result to PDF.
"""

import sys
from docopt import docopt
from pathlib import Path

from dolfin import Mesh, MeshValueCollection, XDMFFile, MeshFunction, plot
from plotstuff import PlottingContext


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["XDMF"] = Path(args["XDMF"])
    args["--pdf"] = Path(args["--pdf"]) if args["--pdf"] else None
    return args


def main(args):
    args = parse_arguments(args)
    mesh = Mesh()
    mvc = MeshValueCollection("size_t", mesh, dim=None)
    with XDMFFile(args["XDMF"].as_posix()) as f:
        f.read(mesh)
        if args["--subdomains"]:
            f.read(mvc, "gmsh:physical")

    plot_argv = [__file__, args["--pdf"]] if args["--pdf"] else [__file__]
    with PlottingContext(plot_argv, "pdiercks_multi") as fig:
        if args["--subdomains"]:
            subdomains = MeshFunction("size_t", mesh, mvc)
            ps = plot(subdomains)
            if args["--colorbar"]:
                fig.colorbar(ps)
        plot(mesh)


if __name__ == "__main__":
    main(sys.argv[1:])
