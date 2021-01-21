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
from numpy import arange, genfromtxt, exp

PREPROCESSING = Path(__file__).parent


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["DATA"] = Path(args["DATA"])
    return args


def main(args):
    args = parse_arguments(args)

    # BAM colors
    with open(PREPROCESSING / "bamcolors_hex.yml", "r") as instream:
        bamcd = yaml.safe_load(instream)
    keys = ["BAMred1", "BAMblue1", "BAMgreen1"]

    with open(args["DATA"], "r") as f:
        header = f.readline()

    names = header.strip("#\n ").split(", ")
    errors = genfromtxt(args["DATA"], delimiter=",")

    plot_argv = [__file__, args["--output"]] if args["--output"] else [__file__]
    with PlottingContext(plot_argv, "pdiercks_multi") as fig:
        ax = fig.subplots()
        for e, n, k in zip(errors.T, names, keys):
            modes = arange(e.size)
            ax.semilogy(
                modes + 1, e, color=bamcd[k]["c"], marker=bamcd[k]["m"], label=n
            )

        # FIXME should be part of DATA to load ...
        reference = exp(-modes / 5)
        ax.semilogy(modes + 1, reference, "k--", label=r"$\exp(-N/5)$")

        ax.set_xlabel(r"Number of modes $N$")
        ylabel = r"\max_j\norm{s_j - \sum_i(s_j, \xi_i)_V \xi_i}_V"
        ax.set_ylabel(r"${}$".format(ylabel))
        ax.legend()


if __name__ == "__main__":
    main(sys.argv[1:])
