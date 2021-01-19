"""
plot relative error against number of modes

Usage:
    plot_error_against_modes.py [options] DEG DATA...

Arguments:
    DEG       Degree of FE space used.
    DATA      The data to be plotted.

Options:
    -h, --help               Show this message.
    --label=LABEL            Type of label (disc or basis_type).
    -o FILE, --output=FILE   Write to pdf.
"""

import sys
import yaml
from pathlib import Path
from docopt import docopt
from plotstuff import PlottingContext
from numpy import load, arange

root = Path(__file__).parent.absolute().parent


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["DEG"] = int(args["DEG"])
    args["DATA"] = [Path(d) for d in args["DATA"]]
    args["--label"] = str(args["--label"]) if args["--label"] else None
    assert all([d.exists() for d in args["DATA"]])
    if len(args["DATA"]) > 3:
        raise NotImplementedError
    return args


def main(args):
    args = parse_arguments(args)

    # BAM colors
    with open(Path(__file__).parent / "bamcolors_hex.yml") as instream:
        bamcd = yaml.safe_load(instream)

    colors = [bamcd["BAMred1"], bamcd["BAMblue2"], bamcd["BAMgreen1"]]
    markers = ["x", "+", "<"]

    plot_argv = [__file__, args["--output"]] if args["--output"] else [__file__]
    with PlottingContext(plot_argv, "pdiercks_multi") as fig:
        ax = fig.subplots()
        for (instream, c, mark) in zip(args["DATA"], colors, markers):
            # expect error_norms_discretization_degree_basistype.npy
            parts = instream.stem.split("_")
            if args["--label"] == "disc":
                label = parts[2]
            elif args["--label"] == "basis_type":
                label = parts[4]
            else:
                label = None
            error = load(instream)
            modes = arange(error.size) + 1
            ax.semilogy(modes, error, color=c, marker=mark, label=label)
        ax.set_xlabel("Number of modes")
        numerator = r"\norm{u_{\mathrm{dns}} - u_{\mathrm{rb}}}"
        denominator = r"\norm{u_{\mathrm{dns}}}"
        ax.set_ylabel(r"$\nicefrac{{{}}}{{{}}}$".format(numerator, denominator))
        if label:
            ax.legend()


if __name__ == "__main__":
    main(sys.argv[1:])
