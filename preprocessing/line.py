"""
get the mesh for a line (in 2d space)

Usage:
    line.py [options] X Y

Arguments:
    X              Start point like '[Xx, Xy]'.
    Y              End point like '[Yx, Yy]'.

Options:
    -h, --help                         Show this message.
    -l LCAR, --lcar=LCAR               Specify characteristic length.
    -t NTRANS, --transfinite=NTRANS    Set transfinite line.
    -o FILE, --output=FILE             Specify output file. [default: ./line.xdmf]
"""

import sys
from docopt import docopt
import pygmsh
import meshio


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["X"] = eval(args["X"])
    args["Y"] = eval(args["Y"])
    args["--output"] = str(args["--output"])
    return args


def main(args):
    args = parse_arguments(args)
    geom = pygmsh.built_in.Geometry()

    if args["--lcar"]:
        lcar = float(args["--lcar"])
    else:
        lcar = None

    X = args["X"]
    Y = args["Y"]
    p0 = geom.add_point([X[0], X[1], 0.0], lcar=lcar)
    p1 = geom.add_point([Y[0], Y[1], 0.0], lcar=lcar)

    l = geom.add_line(p0, p1)
    if args["--transfinite"]:
        N = int(args["--transfinite"])
        geom.set_transfinite_lines([l], size=N)

    # add physical such that mesh only contains one cell type
    geom.add_physical(l, label="line")

    mesh = pygmsh.generate_mesh(geom)
    mesh.prune_z_0()
    meshio.write(args["--output"], meshio.Mesh(points=mesh.points, cells=mesh.cells))


if __name__ == "__main__":
    main(sys.argv[1:])
