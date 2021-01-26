"""
generate mesh for a structured RVE with one inclusion

Usage:
    rve_type_01.py [options] X0 Y0 X Y R N

Arguments:
    X0                  X-coordinate of lower left corner.
    Y0                  Y-coordinate of lower left corner.
    X                   X-coordinate of upper right corner.
    Y                   Y-coordinate of upper right corner.
    R                   Radius of the inclusion.
    N                   Results in 2N - 1 points on each edge.

Options:
    -h, --help                    Show this message.
    -o FILE, --output=FILE        Set output path. [default: ./rve.xdmf]
"""

import sys
import os
from docopt import docopt

import pygmsh
import meshio


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["X0"] = float(args["X0"])
    args["Y0"] = float(args["Y0"])
    args["X"] = float(args["X"])
    args["Y"] = float(args["Y"])
    args["R"] = float(args["R"])
    args["N"] = int(args["N"])
    return args


def add_structured_rve(
    geom,
    X0=0.0,
    Y0=0.0,
    X=1.0,
    Y=1.0,
    R=0.2,
    N=6,
    subdomains=True,
    boundaries=False,
    return_entities=False,
):
    """add a RVE with structured mesh (transfinite) to the geometry object

    Parameters
    ----------
    geom
        An instance of pygmsh.built_in.Geometry().
    X0 : float, optional
        X-coordinate of lower left corner.
    Y0 : float, optional
        Y-coordinate of lower left corner.
    X : float, optional
        X-coordinate of upper right corner.
    Y : float, optional
        Y-coordinate of upper right corner.
    R : float, optional
        Radius of the inclusion.
    N : int, optional
        Results in 2N - 1 points on each edge.
    subdomains : bool, optional
        If True, add physical surface for fenics subdomain marking.
    boundaries : bool, optional
        If True add physical lines for Fenics boundary marking.
    return_entities : bool, optional
        If True return surface and boundary entities.

    """

    def add_transfinite_surface_via_points(geometry, surface_id, points, alternate=""):
        """add a transfinite surfaces by specifying corner points of the surface"""
        code = f"Transfinite Surface {{ {surface_id} }} = "
        code += f"{{ {points[0].id}, {points[1].id}, {points[2].id}, {points[3].id} }} "
        code += alternate
        code += ";"
        geometry.add_raw_code(code)

    def add_rectangle(geom, X0=0.0, Y0=0.0, a=1.0, b=1.0, lcar=0.1):
        """add all points and lines to define a square defined by 8 points

        Parameters
        ----------
        X0
            x-component of lower left corner.
        Y0
            y-component of lower left corner.
        a
            horizontal edge length.
        b
            vertical edge length.
        """
        XX = [
            [X0, Y0, 0.0],
            [X0 + a / 2, Y0, 0.0],
            [X0 + a, Y0, 0.0],
            [X0 + a, Y0 + b / 2, 0.0],
            [X0 + a, Y0 + b, 0.0],
            [X0 + a / 2, Y0 + b, 0.0],
            [X0, Y0 + b, 0.0],
            [X0, Y0 + b / 2, 0.0],
        ]
        LCAR = [
            lcar,
        ] * len(XX)
        p = [geom.add_point(x, lcar=l) for x, l in zip(XX, LCAR)]
        lines = [geom.add_line(p[k], p[k + 1]) for k in range(len(p) - 1)]
        lines.append(geom.add_line(p[-1], p[0]))
        return p, lines

    radius = R
    lcar = 0.1
    #  rotation_matrix = pygmsh.rotation_matrix([0, 0, 1], np.pi * 45 / 180)
    circle = geom.add_circle(
        [X0 + (X - X0) / 2, Y0 + (Y - Y0) / 2, 0.0],
        radius,
        lcar=lcar,
        R=None,
        num_sections=8,
        make_surface=True,
    )
    cl = circle.line_loop.lines

    # transfinite circle
    # 2N - 1 points on e.g. segment cl[5:6]
    geom.set_transfinite_lines(cl, size=N)
    cpoints = [cl[0].start, cl[2].start, cl[4].start, cl[6].start]
    add_transfinite_surface_via_points(
        geom, circle.plane_surface.id, cpoints, alternate="AlternateLeft"
    )

    # square
    sp, sl = add_rectangle(geom, X0=X0, Y0=Y0, a=X - X0, b=Y - Y0)

    # diagonals
    d0 = geom.add_line(sp[0], cl[4].end)
    d1 = geom.add_line(sp[1], cl[5].end)
    d2 = geom.add_line(sp[2], cl[6].end)
    d3 = geom.add_line(sp[3], cl[7].end)
    d4 = geom.add_line(sp[4], cl[0].end)
    d5 = geom.add_line(sp[5], cl[1].end)
    d6 = geom.add_line(sp[6], cl[2].end)
    d7 = geom.add_line(sp[7], cl[3].end)
    geom.set_transfinite_lines([d0, d2, d4, d6], size=N + 2, progression=0.9)
    geom.set_transfinite_lines([d1, d3, d5, d7], size=N + 2, progression=1.0)
    geom.set_transfinite_lines(sl, size=N)

    # line loops
    ll0 = geom.add_line_loop([sl[0], d1, -cl[5], -d0])
    ll1 = geom.add_line_loop([sl[1], d2, -cl[6], -d1])
    ll2 = geom.add_line_loop([sl[2], d3, -cl[7], -d2])
    ll3 = geom.add_line_loop([sl[3], d4, -cl[0], -d3])
    ll4 = geom.add_line_loop([sl[4], d5, -cl[1], -d4])
    ll5 = geom.add_line_loop([sl[5], d6, -cl[2], -d5])
    ll6 = geom.add_line_loop([sl[6], d7, -cl[3], -d6])
    ll7 = geom.add_line_loop([sl[7], d0, -cl[4], -d7])

    # surfaces
    s0 = geom.add_plane_surface(ll0)
    s1 = geom.add_plane_surface(ll1)
    s2 = geom.add_plane_surface(ll2)
    s3 = geom.add_plane_surface(ll3)
    s4 = geom.add_plane_surface(ll4)
    s5 = geom.add_plane_surface(ll5)
    s6 = geom.add_plane_surface(ll6)
    s7 = geom.add_plane_surface(ll7)

    add_transfinite_surface_via_points(
        geom, s0.id, [sp[0], sp[1], cl[5].end, cl[4].end], alternate="Left"
    )
    add_transfinite_surface_via_points(
        geom, s1.id, [sp[1], sp[2], cl[6].end, cl[5].end], alternate="Right"
    )
    add_transfinite_surface_via_points(
        geom, s2.id, [sp[2], sp[3], cl[7].end, cl[6].end], alternate="Left"
    )
    add_transfinite_surface_via_points(
        geom, s3.id, [sp[3], sp[4], cl[0].end, cl[7].end], alternate="Right"
    )
    add_transfinite_surface_via_points(
        geom, s4.id, [sp[4], sp[5], cl[1].end, cl[0].end], alternate="Left"
    )
    add_transfinite_surface_via_points(
        geom, s5.id, [sp[5], sp[6], cl[2].end, cl[1].end], alternate="Right"
    )
    add_transfinite_surface_via_points(
        geom, s6.id, [sp[6], sp[7], cl[3].end, cl[2].end], alternate="Left"
    )
    add_transfinite_surface_via_points(
        geom, s7.id, [sp[7], sp[0], cl[4].end, cl[3].end], alternate="Right"
    )

    # physical surfaces
    if subdomains:
        geom.add_physical([s0, s1, s2, s3, s4, s5, s6, s7], label="matrix")
        geom.add_physical(circle.plane_surface, label="inclusion")

    # physical lines
    if boundaries:
        geom.add_physical(sl[:2], label="bottom")
        geom.add_physical(sl[2:4], label="right")
        geom.add_physical(sl[4:6], label="top")
        geom.add_physical(sl[6:], label="left")

    if return_entities:
        return {
            "matrix": [s0, s1, s2, s3, s4, s5, s6, s7],
            "inclusion": [
                circle.plane_surface,
            ],
            "boundary": sl,
        }


def main(args):
    args = parse_arguments(args)

    path = os.path.dirname(os.path.abspath(args["--output"]))
    base = os.path.splitext(os.path.basename(args["--output"]))[0]
    ext = os.path.splitext(os.path.basename(args["--output"]))[1]

    geom = pygmsh.built_in.Geometry()
    add_structured_rve(
        geom,
        X0=args["X0"],
        Y0=args["Y0"],
        X=args["X"],
        Y=args["Y"],
        R=args["R"],
        N=args["N"],
        subdomains=True,
        boundaries=False,
    )
    geom.add_raw_code("Coherence;")  # removes duplicate points
    geom.add_raw_code("Mesh.Smoothing = 100;")

    mshfile = path + "/" + base + ext if ext == ".msh" else None
    geofile = path + "/" + base + ext if ext == ".geo" else None
    mesh = pygmsh.generate_mesh(
        geom, geo_filename=geofile, msh_filename=mshfile, mesh_file_type="msh2"
    )

    if ext == ".xdmf":
        mesh.prune_z_0()
        meshio.write(
            args["--output"],
            meshio.Mesh(points=mesh.points, cells=mesh.cells, cell_data=mesh.cell_data),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
