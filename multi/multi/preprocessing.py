"""preprocessing module to generate computational grids with Gmsh"""

# TODO replace scripts in /multicode/preprocessing by functions
# TODO test if out_file with format .xdmf can be imported in dolfin

# TODO convenience function write_xdmf() using `create_mesh` as in the fenicsx-tutorial
# see https://jorgensd.github.io/dolfinx-tutorial/chapter3/subdomains.html#convert-msh-files-to-xdmf-using-meshio
# ... TODO need to prune z coordinate for xdmf input
# ... TODO extract cell_data if writing to .xdmf

# TODO [dolfinx] use dolfinx.io.gmshio.model_to_mesh()
# ... or dolfinx.io.gmshio.read_from_msh()
# ... or convert msh to xdmf using meshio and read with dolfinx.io.XDMFFile
# it seems that the latter approach is favorable for parallel IO of the mesh?

import pathlib
import tempfile
import gmsh
import meshio
import numpy as np


def to_array(values):
    floats = []
    try:
        for v in values:
            floats.append(float(v))
    except TypeError:
        floats = [float(values)]

    return np.array(floats, dtype=float)


def _write(filepath, mesh):
    p = pathlib.Path(filepath)
    suffix = p.suffix

    if suffix == ".msh":
        meshio.write(p.as_posix(), mesh, file_format="gmsh")
    elif suffix == ".xdmf":
        meshio.write(p.as_posix(), mesh, file_format="xdmf")
    else:
        raise NotImplementedError


def _generate_and_read_grid(dim):
    gmsh.model.mesh.generate(dim)

    tf = tempfile.NamedTemporaryFile(suffix=".msh", delete=True)
    filepath = tf.name
    gmsh.write(filepath)
    gmsh.finalize()
    grid = meshio.read(filepath)
    tf.close()
    return grid


def create_line_grid(start, end, lc=0.1, num_cells=None, out_file=None):
    """TODO docstring"""
    start = to_array(start)
    end = to_array(end)

    gmsh.initialize()
    gmsh.model.add("line")

    p0 = gmsh.model.geo.addPoint(*start, lc)
    p1 = gmsh.model.geo.addPoint(*end, lc)
    line = gmsh.model.geo.addLine(p0, p1)

    if num_cells is not None:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, num_cells + 1)

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [line])

    grid = _generate_and_read_grid(1)
    if out_file is not None:
        _write(out_file, grid)

    return grid


def create_rectangle_grid(
    xmin,
    xmax,
    ymin,
    ymax,
    z=0.0,
    lc=0.1,
    num_cells=None,
    recombine=False,
    out_file=None,
):
    """TODO docstring"""
    gmsh.initialize()
    gmsh.model.add("rectangle")

    p0 = gmsh.model.geo.addPoint(xmin, ymin, z, lc)
    p1 = gmsh.model.geo.addPoint(xmax, ymin, z, lc)
    p2 = gmsh.model.geo.addPoint(xmax, ymax, z, lc)
    p3 = gmsh.model.geo.addPoint(xmin, ymax, z, lc)

    l0 = gmsh.model.geo.addLine(p0, p1)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p0)

    curve_loop = gmsh.model.geo.addCurveLoop([l0, l1, l2, l3])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    if num_cells is not None:
        try:
            nx, ny = num_cells
        except TypeError:
            nx = ny = num_cells
        gmsh.model.geo.mesh.setTransfiniteCurve(l0, nx + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l2, nx + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l1, ny + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l3, ny + 1)

        gmsh.model.geo.mesh.setTransfiniteSurface(surface, "Left")

        if recombine:
            # setRecombine(dim, tag, angle=45.0)
            gmsh.model.geo.mesh.setRecombine(2, surface)

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [surface])

    grid = _generate_and_read_grid(2)
    if out_file is not None:
        _write(out_file, grid)

    return grid


def create_rce_grid_01(xmin, xmax, ymin, ymax, z=0.0, radius=0.2, lc=0.1, num_cells_per_edge=None):
    """TODO docstring"""
    gmsh.initialize()
    gmsh.model.add("rce_01")

    geom = gmsh.model.geo

    # add the inclusion (circle) as 8 circle arcs
    phi = np.linspace(0, 2*np.pi, num=8, endpoint=True)
    x_center = np.tile(np.array([xmin + (xmax-xmin)/2, ymin + (ymax-ymin)/2, z]), (8, 1))
    x_unit_circle = np.array([np.cos(phi), np.sin(phi), np.zeros_like(phi)]).T
    x_circle = x_center + x_unit_circle

    center = geom.add_point(*xc[0], lc)

    circle_points = []
    for xyz in x_circle:
        p = geom.add_point(*xyz, lc)
        cirle_points.append(p)

    breakpoint()
    # gmsh.model.geo.add_circle_arc(startTag, centerTag, endTag)
    # all lines should be transfinite

    # add the rectangle defined by 8 points

    # add diagonal lines between points of the rectangle and the circle

    # set diagonal lines as transfinite

    # add the 8 line loops (surfaces between circle and rectangle)
    # add surfaces accordingly

    # set surfaces as transfinite (requires points of the surfaces)

    # set physical groups for the matrix and inclusion
