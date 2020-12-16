"""
create a partition of a rectangle

Usage:
    rectangle.py [options] X Y NX NY LCAR

Arguments:
    X                 x coordinate of upper right corner.
    Y                 y coordinate of upper right cornder.
    NX                Number of elements in x-direction.
    NY                Number of elements in y-direction.
    LCAR              Characteristic length.

Options:
    -h, --help                    Show this message.
    --transfinite                 Set transifnite surface, which results in a structured mesh.
    --quads                       Recombine triangles to quadrilaterals.
    --offx=OFFX                   Offset (float) for x coordinate of origin [default: 0.0].
    --offy=OFFY                   Offset (float) for y coordinate of origin [default: 0.0].
    --SecondOrderIncomplete=INT   Create incomplete second order elements? [default: 0].
    -o FILE --output=FILE         Specify output file (msh or vtu). [default: ./rectangle.msh]
"""

import os
import sys

import pygmsh
from docopt import docopt


def parse_arguments(args):
    args = docopt(__doc__, args)
    args['X'] = float(args['X'])
    args['Y'] = float(args['Y'])
    args['NX'] = int(args['NX'])
    args['NY'] = int(args['NY'])
    args['LCAR'] = float(args['LCAR'])
    args['--offx'] = float(args['--offx'])
    args['--offy'] = float(args['--offy'])
    args["--SecondOrderIncomplete"] = int(args["--SecondOrderIncomplete"])
    return args


def generate_mesh(args):
    path = os.path.dirname(os.path.abspath(args['--output']))
    base = os.path.splitext(os.path.basename(args['--output']))[0]
    ext = os.path.splitext(os.path.basename(args['--output']))[1]

    NX = args['NX']
    NY = args['NY']
    LCAR = args['LCAR']

    X = args['X']
    Y = args['Y']
    XO = args['--offx']
    YO = args['--offy']

    geom = pygmsh.built_in.Geometry()
    geom.add_raw_code(f"Mesh.SecondOrderIncomplete = {args['--SecondOrderIncomplete']};")
    square = geom.add_polygon(
        [[XO, YO, 0.0], [X, YO, 0.0], [X, Y, 0.0], [XO, Y, 0.0]], LCAR
    )

    if args['--transfinite']:
        geom.set_transfinite_surface(square.surface, size=[NX + 1, NY + 1])
    if args['--quads']:
        geom.add_raw_code("Recombine Surface {%s};" % square.surface.id)

    geom.add_physical([square.surface], label="surface")

    mshfile = path + '/' + base + ext if ext == '.msh' else None
    geofile = path + '/' + base + ext if ext == '.geo' else None
    mesh = pygmsh.generate_mesh(geom, geo_filename=geofile, msh_filename=mshfile, prune_z_0=True)

    if ext == '.vtu':
        import meshio
        meshio.write(path + '/' + base + ext, mesh)


def main(args):
    args = parse_arguments(args)
    generate_mesh(args)


if __name__ == "__main__":
    main(sys.argv[1:])
