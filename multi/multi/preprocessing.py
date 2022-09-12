"""preprocessing module to generate computational grids with Gmsh"""

# TODO replace scripts in /multicode/preprocessing by functions

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
        gmsh.model.geo.mesh.setTransfiniteCurve(line, num_cells+1)

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [line])

    grid = _generate_and_write_grid(1, out_file)
    return grid


def create_rectangle_grid(start, end, lc=0.1, num_cells=None, out_file=None):
    """TODO docstring"""
    start = to_array(start)
    end = to_array(end)
    dist = np.abs(end - start)
    # TODO generalize rectangle in 3d space, start & end are not enough?

    gmsh.initialize()
    gmsh.model.add("rectangle")

    from IPython import embed;embed()
