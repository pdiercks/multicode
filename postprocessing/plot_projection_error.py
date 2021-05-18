"""
plot projection error against number of modes

Usage:
    plot_projection_error.py [options] DATA

Arguments:
    DATA      FilePath (incl. .txt).

Options:
    -h, --help               Show this message.
    -o FILE, --output=FILE   Write to pdf.
"""

import sys
import yaml
from pathlib import Path
from docopt import docopt
from plotstuff import PlottingContext
from matplotlib.lines import Line2D
from numpy import arange, genfromtxt, exp

POSTPROCESSING = Path(__file__).parent


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["DATA"] = Path(args["DATA"])
    return args


def main(args):
    args = parse_arguments(args)

    # BAM colors
    with open(POSTPROCESSING / "bamcolors_rgb.yml", "r") as instream:
        bamcd = yaml.safe_load(instream)

    red = bamcd["red"]
    blue = bamcd["blue"]
    green = bamcd["green"]
    yellow = bamcd["yellow"]
    black = bamcd["black"]

    colors = []
    for r, b, g, y in zip(red, blue, green, yellow):
        colors.append(tuple(r))
        colors.append(tuple(b))
        colors.append(tuple(g))
        colors.append(tuple(y))

    markers_dict = Line2D.markers
    markers_dict.pop(",")  # do not like this one
    markers = list(markers_dict)

    with open(args["DATA"], "r") as f:
        header = f.readline()

    names = header.strip("#\n ").split(", ")
    errors = genfromtxt(args["DATA"], delimiter=",")

    plot_argv = [__file__, args["--output"]] if args["--output"] else [__file__]
    with PlottingContext(plot_argv, "pdiercks_multi") as fig:
        ax = fig.subplots()
        for k, (e, n) in enumerate(zip(errors.T, names)):
            modes = arange(e.size)
            ax.semilogy(modes, e, color=colors[k], marker=markers[k], label=n)

        # FIXME should be part of DATA to load ...
        reference = exp(-modes / 5)
        ax.semilogy(modes, reference, color=black[0], ls="--", label=r"$\exp(-N/5)$")

        ax.set_xlabel(r"Number of modes $N$")
        ylabel = r"\max_j\norm{s_j - \sum_i(s_j, \xi_i)_V \xi_i}_V"
        ax.set_ylabel(r"${}$".format(ylabel))
        ax.legend()


if __name__ == "__main__":
    main(sys.argv[1:])
