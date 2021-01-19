"""
plot relative error against number of modes.

Usage:
    plot_error_against_modes.py [options] DATA...
    plot_error_against_modes.py [options] DATA... [-l LABEL]...

Arguments:
    DATA      The data (incl. ext .npy) to be plotted.

Options:
    -h, --help               Show this message.
    -l, --label=LABEL        Add a label for each data set.
    -o FILE, --output=FILE   Write to pdf.
"""

import sys
import yaml
from pathlib import Path
from docopt import docopt
from plotstuff import PlottingContext
from numpy import load, arange

POSTPROCESSING = Path(__file__).parent


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["DATA"] = [Path(d) for d in args["DATA"]]
    assert all([d.exists() for d in args["DATA"]])
    assert all([d.suffix == ".npy" for d in args["DATA"]])

    if args["--label"]:
        args["--label"] = [str(lbl) for lbl in args["--label"]]
        assert len(args["--label"]) == len(args["DATA"])
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

    cc = ["blue", "green", "red"]
    keys = []
    for i in range(1, 4):
        for c in cc:
            s = "BAM" + c + f"{i}"
            keys.append(s)

    if len(args["DATA"]) > len(keys):
        raise NotImplementedError

    plot_argv = [__file__, args["--output"]] if args["--output"] else [__file__]
    with PlottingContext(plot_argv, "pdiercks_multi") as fig:
        ax = fig.subplots()
        for (i, f) in enumerate(args["DATA"]):
            error = load(f)
            modes = arange(error.size) + 1
            ax.semilogy(
                modes,
                error,
                color=bamcd[keys[i]]["c"],
                marker=bamcd[keys[i]]["m"],
                label=args["--label"][i],
            )
        ax.set_xlabel("Number of modes")
        numerator = r"\norm{u_{\mathrm{dns}} - u_{\mathrm{rb}}}"
        denominator = r"\norm{u_{\mathrm{dns}}}"
        ax.set_ylabel(r"$\nicefrac{{{}}}{{{}}}$".format(numerator, denominator))
        if args["legend"]:
            ax.legend()


if __name__ == "__main__":
    main(sys.argv[1:])
