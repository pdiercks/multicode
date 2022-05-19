"""
generate edge meshes for given RCE mesh

Usage:
    rce_edges.py [options] RCE

Arguments:
    RCE       Filepath (incl. .xdmf extension).

Options:
    -h, --help               Show this message and exit.
    --transfinite=N          Use N equidistant points on each edge.
    --lcar=LCAR              Specify characteristic length.
    -o FILE, --output=FILE   Optional filepath such that edge meshes
                             are written as FILE_{edge}.xdmf. Otherwise
                             FILE=Path(RVE).
"""

import sys
from pathlib import Path
from docopt import docopt

import meshio
import pygmsh

from multi import Domain


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["RVE"] = Path(args["RVE"])
    return args


def main(args):
    args = parse_arguments(args)
    domain = Domain(args["RVE"])
    points = {
        "bottom": ([domain.xmin, domain.ymin], [domain.xmax, domain.ymin]),
        "right": ([domain.xmax, domain.ymin], [domain.xmax, domain.ymax]),
        "top": ([domain.xmin, domain.ymax], [domain.xmax, domain.ymax]),
        "left": ([domain.xmin, domain.ymin], [domain.xmin, domain.ymax]),
    }
    for edge, (start, end) in points.items():
        create_edge_mesh(args, start, end, edge)


def create_edge_mesh(args, X, Y, edge):
    """create partition of a line in two space dimensions

    Parameters
    ----------
    X : tuple, list of floats
        Start point of the line.
    Y : tuple, list of floats
        End point of the line.
    edge : str
        The edge label.
    """
    geom = pygmsh.built_in.Geometry()

    if args["--lcar"]:
        lcar = float(args["--lcar"])
    else:
        lcar = None

    p0 = geom.add_point([X[0], X[1], 0.0], lcar=lcar)
    p1 = geom.add_point([Y[0], Y[1], 0.0], lcar=lcar)

    line = geom.add_line(p0, p1)
    if args["--transfinite"]:
        N = int(args["--transfinite"])
        geom.set_transfinite_lines([line], size=N)

    # add physical such that mesh only contains one cell type
    geom.add_physical(line, label="line")

    mesh = pygmsh.generate_mesh(geom)
    mesh.prune_z_0()
    if args["--output"]:
        xdmf = args["--output"].parent / (args["--output"].stem + f"_{edge}.xdmf")
    else:
        xdmf = args["RVE"].parent / (args["RVE"].stem + f"_{edge}.xdmf")
    meshio.write(xdmf, meshio.Mesh(points=mesh.points, cells=mesh.cells))


if __name__ == "__main__":
    main(sys.argv[1:])
