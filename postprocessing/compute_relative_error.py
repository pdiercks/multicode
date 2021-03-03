"""
compute global relative error

Usage:
    compute_relative_error.py [options] NORMS

Arguments:
    NORMS        TODO

Options:
    -h, --help     Show this message and exit.
    --output=NPY   Save as NPY.
"""

import sys
from pathlib import Path
from docopt import docopt

import numpy as np


def parse_args(args):
    args = docopt(__doc__, args)
    args["NORMS"] = Path(args["NORMS"])
    return args


def main(args):
    args = parse_args(args)

    n = np.load(args["NORMS"])
    assert all([k in ("dns", "err") for k in n.files])
    (modes, ncells, _) = n["dns"].shape
    err = n["err"].reshape(modes, ncells)
    dns = n["dns"].reshape(modes, ncells)
    err_gl = np.sqrt(np.sum(err ** 2, axis=1))
    dns_gl = np.sqrt(np.sum(dns ** 2, axis=1))
    rel = err_gl / dns_gl
    if args["--output"] is not None:
        np.save(Path(args["--output"]), rel)


if __name__ == "__main__":
    main(sys.argv[1:])
