"""preprocessing module to generate computational grids with Gmsh"""

# TODO replace scripts in /multicode/preprocessing by functions

import pathlib
import tempfile
import gmsh
import meshio


def to_floats(values):
    floats = []
    try:
        for v in values:
            floats.append(float(v))
    except TypeError:
        floats = [float(values)]

    return floats


def create_line_grid(start, end, lc=0.1, num_cells=None, out_file=None):
    """TODO docstring"""
    start = to_floats(start)
    end = to_floats(end)

    # # adjust the values such that start < end for all dimensions
    # assert len(start) == len(end)
    # for i in range(len(start)):
    #     if start[i] > end[i]:
    #         start[i], end[i] = end[i], start[i]

    gmsh.initialize()
    gmsh.model.add("line")

    p0 = gmsh.model.geo.addPoint(*start, lc)
    p1 = gmsh.model.geo.addPoint(*end, lc)
    line = gmsh.model.geo.addLine(p0, p1)
    gmsh.model.addPhysicalGroup(1, [line])

    if num_cells is not None:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, num_cells+1)

    gmsh.model.mesh.generate(1)

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

