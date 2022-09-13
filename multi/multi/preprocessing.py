"""preprocessing module to generate computational grids with Gmsh"""

# TODO replace scripts in /multicode/preprocessing by functions

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


def _generate_and_write_grid(dim, out_file):
    gmsh.model.mesh.generate(dim)

    tf = tempfile.NamedTemporaryFile(suffix=".msh", delete=True)
    if out_file is not None:
        filepath = out_file
    else:
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

    grid = _generate_and_write_grid(1, out_file)
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

    grid = _generate_and_write_grid(2, out_file)
    return grid
