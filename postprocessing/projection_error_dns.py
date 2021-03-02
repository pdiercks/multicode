"""
compute projection error for basis wrt dns

Usage:
    projection_error_dns.py [options] FINE DNS COARSE RVE DEG

Arguments:
    FINE         TODO        
    DNS          TODO        
    COARSE       TODO        
    RVE          TODO        
    DEG          TODO        

Options:
    -h, --help     Show this message and exit.
    --output=TXT   Write projection error to FilePath.
"""

import sys
from pathlib import Path
from docopt import docopt

import dolfin as df
import numpy as np
from multi import Domain, ResultFile, DofMap

from pymor.core.logger import getLogger


def parse_args(args):
    args = docopt(__doc__, args)
    args["FINE"] = Path(args["FINE"])
    args["DNS"] = Path(args["DNS"])
    args["COARSE"] = Path(args["COARSE"])
    args["RVE"] = Path(args["RVE"])
    args["DEG"] = int(args["DEG"])
    args["--output"] = Path(args["--output"]) if args["--output"] is not None else None
    return args


def read_dns(args, logger):
    domain = Domain(args["FINE"], id_=0, subdomains=True)
    V = df.VectorFunctionSpace(domain.mesh, "CG", args["DEG"])
    d = df.Function(V)

    logger.info(f"Reading DNS from file {args['DNS']} ...")
    xdmf = ResultFile(args["DNS"])
    xdmf.read_checkpoint(d, "u-dns", 0)
    xdmf.close()
    return d


def get_snapshots(args, logger, dns, source):
    dofmap = DofMap(args["COARSE"], 2, 2)
    snaps = source.empty(reserve=len(dofmap.cells))
    for cell_index, cell in enumerate(dofmap.cells):
        offset = np.around(dofmap.points[cell][0], decimals=5)
        omega = Domain(
            args["RVE"], id_=cell_index, subdomains=True, translate=df.Point(offset)
        )
        V = df.VectorFunctionSpace(omega.mesh, "CG", args["DEG"])
        Idns = df.interpolate(dns, V)
        snaps.append(source.make_array([Idns.vector()]))


def main(args):
    args = parse_args(args)
    logger = getLogger("main")
    dns = read_dns(args, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
