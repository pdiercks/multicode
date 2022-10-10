"""preprocessing module to generate computational grids with Gmsh"""

# Usage
#######
# Use Gmsh python API to generate a grid and write to .msh.
# the mesh and optional cell markers can be read with dolfinx.io.gmshio in serial.
# For parallel input of meshes, first convert .msh to .xdmf using meshio
# and read the mesh with dolfinx.io.XDMFFile.

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


# this code is part of the FEniCSx tutorial
# see https://jorgensd.github.io/dolfinx-tutorial/chapter3/subdomains.html#convert-msh-files-to-xdmf-using-meshio
def create_mesh(mesh, cell_type, prune_z=False, name_to_read="gmsh:physical"):
    """create a new instance of meshio.Mesh from meshio.Mesh object"""
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data(name_to_read, cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
            points=points, cells={cell_type: cells}, cell_data={name_to_read:[cell_data]}
            )
    return out_mesh


def _generate_and_write_grid(dim, filepath):
    gmsh.model.mesh.generate(dim)
    gmsh.write(filepath)
    gmsh.finalize()


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

    filepath = out_file or "./line.msh"
    _generate_and_write_grid(1, filepath)


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

    filepath = out_file or "./rectangle.msh"
    _generate_and_write_grid(2, filepath)


def create_rce_grid_01(xmin, xmax, ymin, ymax, z=0.0, radius=0.2, lc=0.1, num_cells_per_edge=None, out_file=None):
    """TODO docstring"""

    width = abs(xmax-xmin)
    height = abs(ymax-ymin)

    gmsh.initialize()
    gmsh.model.add("rce_01")

    # options
    gmsh.option.setNumber("Mesh.Smoothing", 2)

    geom = gmsh.model.geo

    # add the inclusion (circle) as 8 circle arcs
    phi = np.linspace(0, 2*np.pi, num=9, endpoint=True)[:-1]
    x_center = np.array([xmin + width/2, ymin + height/2, z])
    x_unit_circle = np.array([radius * np.cos(phi), radius * np.sin(phi), np.zeros_like(phi)]).T
    x_circle = np.tile(x_center, (8, 1)) + x_unit_circle

    center = geom.add_point(*x_center, lc)

    circle_points = []
    for xyz in x_circle:
        p = geom.add_point(*xyz, lc)
        circle_points.append(p)

    circle_arcs = []
    for i in range(len(circle_points)-1):
        arc = geom.add_circle_arc(circle_points[i], center, circle_points[i+1])
        circle_arcs.append(arc)
    arc = geom.add_circle_arc(circle_points[-1], center, circle_points[0])
    circle_arcs.append(arc)
    circle_loop = geom.add_curve_loop(circle_arcs)

    circle_surface = geom.add_plane_surface([circle_loop])

    # add the rectangle defined by 8 points
    dx = np.array([width/2, 0., 0.])
    dy = np.array([0., height/2, 0.])
    x_rectangle = np.stack([
        x_center + dx,
        x_center + dx + dy,
        x_center + dy,
        x_center - dx + dy,
        x_center - dx,
        x_center - dx - dy,
        x_center - dy,
        x_center + dx - dy,
        ])

    rectangle_points = []
    for xyz in x_rectangle:
        p = geom.add_point(*xyz, lc)
        rectangle_points.append(p)

    # draw rectangle lines
    rectangle_lines = []
    for i in range(len(rectangle_points)-1):
        line = geom.add_line(rectangle_points[i], rectangle_points[i+1])
        rectangle_lines.append(line)
    line = geom.add_line(rectangle_points[-1], rectangle_points[0])
    rectangle_lines.append(line)

    # connect rectangle points and circle points from outer to inner
    conn = []
    for i in range(len(circle_points)):
        l = geom.add_line(rectangle_points[i], circle_points[i])
        conn.append(l)

    # add curve loops defining surfaces of the matrix
    mat_loops = []
    for i in range(len(circle_points)-1):
        cloop = geom.add_curve_loop([rectangle_lines[i], conn[i+1], -circle_arcs[i], -conn[i]])
        mat_loops.append(cloop)
    cloop = geom.add_curve_loop([rectangle_lines[-1], conn[0], -circle_arcs[-1], -conn[-1]])
    mat_loops.append(cloop)

    matrix = []
    for curve_loop in mat_loops:
        mat_surface = geom.add_plane_surface([curve_loop])
        matrix.append(mat_surface)

    if num_cells_per_edge is not None:
        if not num_cells_per_edge % 2 == 0:
            raise ValueError("Number of cells per edge must be even for transfinite mesh. Sorry!")

        N = int(num_cells_per_edge) // 2  # num_cells_per_segment

        for line in circle_arcs:
            geom.mesh.set_transfinite_curve(line, N+1)
        for line in rectangle_lines:
            geom.mesh.set_transfinite_curve(line, N+1)
        # diagonal connections
        for line in conn[0::2]:
            geom.mesh.set_transfinite_curve(line, N+3, meshType='Progression', coef=1.0)
        # horizontal or vertical connections
        for line in conn[1::2]:
            geom.mesh.set_transfinite_curve(line, N+3, meshType='Progression', coef=0.9)

        # transfinite surfaces (circle and matrix)
        # geom.mesh.set_transfinite_surface(tag, arrangement='Left', cornerTags=[])
        geom.mesh.set_transfinite_surface(circle_surface, arrangement="AlternateLeft", cornerTags=circle_points[0::2])
        for surface in matrix[0::2]:
            geom.mesh.set_transfinite_surface(surface, arrangement="Right")
        for surface in matrix[1::2]:
            geom.mesh.set_transfinite_surface(surface, arrangement="Left")

    geom.synchronize()
    geom.removeAllDuplicates()

    # physical groups
    gmsh.model.add_physical_group(2, matrix, name="matrix")
    gmsh.model.add_physical_group(2, [circle_surface], name="inclusion")

    filepath = out_file or "./rce_type_01.msh"
    _generate_and_write_grid(2, filepath)
