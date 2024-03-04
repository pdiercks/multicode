"""preprocessing module to generate computational grids with Gmsh"""

# Usage
#######
# Use Gmsh python API to generate a grid and write to .msh.
# the mesh and optional cell markers can be read with dolfinx.io.gmshio in serial.
# For parallel input of meshes, first convert .msh to .xdmf using meshio
# and read the mesh with dolfinx.io.XDMFFile.

import dolfinx
import gmsh
import meshio
import numpy as np
from typing import Optional, Sequence, Callable, Union, Iterable
from multi.misc import to_floats

GMSH_VERBOSITY = 0


# this code is part of the FEniCSx tutorial
# see https://jorgensd.github.io/dolfinx-tutorial/chapter3/subdomains.html#convert-msh-files-to-xdmf-using-meshio
def create_mesh(mesh, cell_type, prune_z=False, name_to_read="gmsh:physical"):
    """create a new instance of meshio.Mesh from meshio.Mesh object"""
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data(name_to_read, cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points, cells={cell_type: cells}, cell_data={name_to_read: [cell_data]}
    )
    return out_mesh


def create_meshtags(
        domain: dolfinx.mesh.Mesh, entity_dim: int, markers: dict[str, tuple[int, Callable]]
) -> tuple[dolfinx.mesh.MeshTags, dict[str, int]]:
    """Creates meshtags for the given markers.

    This code is part of the FEniCSx tutorial
    by Jørgen S. Dokken.
    See https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html?highlight=sorted_facets#implementation # noqa: E501

    Args:
        domain: The computational domain.
        entity_dim: Dimension of the entities to mark.
        markers: The definition of subdomains or boundaries where each key is a string
          and each value is a tuple of an integer and a marker function.

    """
    tdim = domain.topology.dim
    assert entity_dim in (tdim, tdim - 1)

    entity_indices, entity_markers = [], []
    edim = entity_dim
    marked = {}
    for key, (marker, locator) in markers.items():
        entities = dolfinx.mesh.locate_entities(domain, edim, locator)
        entity_indices.append(entities)
        entity_markers.append(np.full_like(entities, marker))
        if entities.size > 0:
            marked[key] = marker
    entity_indices = np.hstack(entity_indices).astype(np.int32)
    entity_markers = np.hstack(entity_markers).astype(np.int32)
    sorted_facets = np.argsort(entity_indices)
    mesh_tags = dolfinx.mesh.meshtags(domain, edim, entity_indices[sorted_facets], entity_markers[sorted_facets])
    return mesh_tags, marked


def _initialize():
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)  # silent except for fatal errors


def _set_gmsh_options(options):
    if options is not None:
        # Example: {'Mesh.ElementOrder': 2, 'Mesh.SecondOrderIncomplete': 1}
        # will result in 'quad8' cell type for quadrilateral mesh
        for key, value in options.items():
            gmsh.option.setNumber(key, value)

def _generate_and_write_grid(dim, filepath):
    gmsh.model.mesh.generate(dim)
    gmsh.write(filepath)
    gmsh.finalize()


def create_line(start: Iterable[float], end: Iterable[float], lc: float = 0.1, num_cells: Optional[int] = None, out_file: Optional[str] = None, options: Optional[dict[str, int]] = None):
    """Creates a grid of a line.

    Args:
        start: Coordinate.
        end: Coordinate.
        lc: Characteristic length of cells.
        num_cells: If not None, a structured mesh with `num_cells` will be created.
        out_file: Write mesh to `out_file`. Default: './rectangle.msh'
        options: Options to pass to GMSH. See https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-options

    """
    start = np.array(to_floats(start))
    end = np.array(to_floats(end))

    _initialize()
    _set_gmsh_options(options)
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


def create_rectangle(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    z: float = 0.0,
    lc: float = 0.1,
    num_cells: Optional[Union[int, Sequence[int]]] = None,
    recombine: bool = False,
    facets: bool = False,
    out_file: Optional[str] = None,
    options: Optional[dict[str, int]] = None,
):
    """Creates a grid of a rectangle.

    Args:
        xmin: Coordinate.
        xmax: Coordinate.
        ymin: Coordinate.
        ymax: Coordinate.
        z: Coordinate.
        lc: Characteristic length of cells.
        num_cells: If not None, a structured mesh with `num_cells` per edge will
        be created.
        recombine: If True, recombine triangles to quadrilaterals.
        facets: If True, create physical groups for facets.
        out_file: Write mesh to `out_file`. Default: './rectangle.msh'
        options: Options to pass to GMSH. See https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-options

    """
    _initialize()
    _set_gmsh_options(options)
    gmsh.model.add("rectangle")

    p0 = gmsh.model.geo.addPoint(xmin, ymin, z, lc)
    p1 = gmsh.model.geo.addPoint(xmax, ymin, z, lc)
    p2 = gmsh.model.geo.addPoint(xmax, ymax, z, lc)
    p3 = gmsh.model.geo.addPoint(xmin, ymax, z, lc)

    l0 = gmsh.model.geo.addLine(p0, p1)  # bottom
    l1 = gmsh.model.geo.addLine(p1, p2)  # right
    l2 = gmsh.model.geo.addLine(p2, p3)  # top
    l3 = gmsh.model.geo.addLine(p3, p0)  # left

    curve_loop = gmsh.model.geo.addCurveLoop([l0, l1, l2, l3])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    if num_cells is not None:
        if isinstance(num_cells, int):
            nx = ny = num_cells
        else:
            nx, ny = num_cells[:2]

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

    # markers for the facets following ordering of
    # entities of multi.dofmap.QuadrilateralDofLayout
    # bottom: 1, left: 2, right: 3, top: 4
    if facets:
        gmsh.model.add_physical_group(1, [l0], 1, name="bottom")
        gmsh.model.add_physical_group(1, [l3], 2, name="left")
        gmsh.model.add_physical_group(1, [l1], 3, name="right")
        gmsh.model.add_physical_group(1, [l2], 4, name="top")

    filepath = out_file or "./rectangle.msh"
    _generate_and_write_grid(2, filepath)


def create_voided_rectangle(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    z: float = 0.0,
    radius: float = 0.2,
    lc: float = 0.1,
    num_cells: Optional[int] = None,
    recombine: bool = False,
    facets: bool = True,
    out_file: Optional[str] = None,
    options: Optional[dict[str, int]] = None,
):
    """Create grid of rectangular unit cell with single circular void.

    Args:
        xmin: Coordinate.
        xmax: Coordinate.
        ymin: Coordinate.
        ymax: Coordinate.
        z: Coordinate.
        radius: Radius of the void.
        lc: Characteristic length of cells.
        num_cells: If not None, a structured mesh with `num_cells` per edge will
        be created. `num_cells` must be even.
        recombine: If True, recombine triangles to quadrilaterals.
        facets: If True, create physical groups for facets.
        out_file: Write mesh to `out_file`. Default: './voided_rectangle.msh'
        options: Options to pass to GMSH. See https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-options

    """

    width = abs(xmax - xmin)
    height = abs(ymax - ymin)

    _initialize()
    _set_gmsh_options(options)
    gmsh.model.add("voided_rectangle")

    # options
    gmsh.option.setNumber("Mesh.Smoothing", 2)

    geom = gmsh.model.geo

    # add the void (circle) as 8 circle arcs
    phi = np.linspace(0, 2 * np.pi, num=9, endpoint=True)[:-1]
    x_center = np.array([xmin + width / 2, ymin + height / 2, z])
    x_unit_circle = np.array(
        [radius * np.cos(phi), radius * np.sin(phi), np.zeros_like(phi)]
    ).T
    x_circle = np.tile(x_center, (8, 1)) + x_unit_circle

    center = geom.add_point(*x_center, lc)

    circle_points = []
    for xyz in x_circle:
        p = geom.add_point(*xyz, lc)
        circle_points.append(p)

    circle_arcs = []
    for i in range(len(circle_points) - 1):
        arc = geom.add_circle_arc(circle_points[i], center, circle_points[i + 1])
        circle_arcs.append(arc)
    arc = geom.add_circle_arc(circle_points[-1], center, circle_points[0])
    circle_arcs.append(arc)

    # add the rectangle defined by 8 points
    dx = np.array([width / 2, 0.0, 0.0])
    dy = np.array([0.0, height / 2, 0.0])
    x_rectangle = np.stack(
        [
            x_center + dx,
            x_center + dx + dy,
            x_center + dy,
            x_center - dx + dy,
            x_center - dx,
            x_center - dx - dy,
            x_center - dy,
            x_center + dx - dy,
        ]
    )

    rectangle_points = []
    for xyz in x_rectangle:
        p = geom.add_point(*xyz, lc)
        rectangle_points.append(p)

    # draw rectangle lines
    rectangle_lines = []
    for i in range(len(rectangle_points) - 1):
        line = geom.add_line(rectangle_points[i], rectangle_points[i + 1])
        rectangle_lines.append(line)
    line = geom.add_line(rectangle_points[-1], rectangle_points[0])
    rectangle_lines.append(line)

    # connect rectangle points and circle points from outer to inner
    conn = []
    for i in range(len(circle_points)):
        line = geom.add_line(rectangle_points[i], circle_points[i])
        conn.append(line)

    # add curve loops defining surfaces of the matrix
    mat_loops = []
    for i in range(len(circle_points) - 1):
        cloop = geom.add_curve_loop(
            [rectangle_lines[i], conn[i + 1], -circle_arcs[i], -conn[i]]
        )
        mat_loops.append(cloop)
    cloop = geom.add_curve_loop(
        [rectangle_lines[-1], conn[0], -circle_arcs[-1], -conn[-1]]
    )
    mat_loops.append(cloop)

    matrix = []
    for curve_loop in mat_loops:
        mat_surface = geom.add_plane_surface([curve_loop])
        matrix.append(mat_surface)

    if num_cells is not None:
        if not num_cells % 2 == 0:
            raise ValueError(
                "Number of cells per edge must be even for transfinite mesh. Sorry!"
            )

        N = int(num_cells) // 2  # num_cells_per_segment

        for line in circle_arcs:
            geom.mesh.set_transfinite_curve(line, N + 1)
        for line in rectangle_lines:
            geom.mesh.set_transfinite_curve(line, N + 1)
        # diagonal connections
        for line in conn[0::2]:
            geom.mesh.set_transfinite_curve(
                line, N + 3, meshType="Progression", coef=1.0
            )
        # horizontal or vertical connections
        for line in conn[1::2]:
            geom.mesh.set_transfinite_curve(
                line, N + 3, meshType="Progression", coef=0.9
            )

        # transfinite surfaces (circle and matrix)
        # geom.mesh.set_transfinite_surface(tag, arrangement='Left', cornerTags=[])
        for surface in matrix[0::2]:
            geom.mesh.set_transfinite_surface(surface, arrangement="Right")
        for surface in matrix[1::2]:
            geom.mesh.set_transfinite_surface(surface, arrangement="Left")

        if recombine:
            for surface in matrix:
                gmsh.model.geo.mesh.setRecombine(2, surface)

    geom.synchronize()
    geom.removeAllDuplicates()

    # ### physical groups
    gmsh.model.add_physical_group(2, matrix, 1, name="matrix")

    # markers for the facets following ordering of
    # entities of multi.dofmap.QuadrilateralDofLayout
    # bottom: 1, left: 2, right: 3, top: 4
    if facets:
        gmsh.model.add_physical_group(1, rectangle_lines[5:7], 1, name="bottom")
        gmsh.model.add_physical_group(1, rectangle_lines[3:5], 2, name="left")
        gmsh.model.add_physical_group(
            1, [rectangle_lines[0], rectangle_lines[-1]], 3, name="right"
        )
        gmsh.model.add_physical_group(1, rectangle_lines[1:3], 4, name="top")

    filepath = out_file or "./voided_rectangle.msh"
    _generate_and_write_grid(2, filepath)


def create_unit_cell_01(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    z: float = 0.0,
    radius: float = 0.2,
    lc: float = 0.1,
    num_cells: Optional[int] = None,
    recombine: bool = False,
    cell_tags: Optional[dict[str, int]] = None,
    facet_tags: Optional[dict[str, int]] = None,
    out_file: Optional[str] = None,
    tag_counter: Optional[dict[int, int]] = None,
    options: Optional[dict[str, int]] = None,
):
    """Create grid of unit cell with single circular inclusion.

    Args:
        xmin: Coordinate.
        xmax: Coordinate.
        ymin: Coordinate.
        ymax: Coordinate.
        z: Coordinate.
        radius: Radius of the inclusion.
        lc: Characteristic length of cells.
        num_cells: If not None, a structured mesh with `num_cells` per edge will
        be created. `num_cells` must be even.
        recombine: If True, recombine triangles to quadrilaterals.
        cell_tags: If not None, specify cell tags as values for keys `matrix` and `inclusion`.
        facet_tags: If not None, specify facet tags as values for keys `bottom`, `left`, `right` and `top`.
        out_file: Write mesh to `out_file`. Default: './unit_cell_01.msh'
        tag_counter: Can be used to keep track of entity `tags`. The key is the topological dimension
        of the entities to keep track of and the value is the counter.
        options: Options to pass to GMSH. See https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-options

    """

    width = abs(xmax - xmin)
    height = abs(ymax - ymin)

    _initialize()
    _set_gmsh_options(options)
    gmsh.model.add("rce_01")

    # options
    gmsh.option.setNumber("Mesh.Smoothing", 2)

    geom = gmsh.model.geo

    # add the inclusion (circle) as 8 circle arcs
    phi = np.linspace(0, 2 * np.pi, num=9, endpoint=True)[:-1]
    x_center = np.array([xmin + width / 2, ymin + height / 2, z])
    x_unit_circle = np.array(
        [radius * np.cos(phi), radius * np.sin(phi), np.zeros_like(phi)]
    ).T
    x_circle = np.tile(x_center, (8, 1)) + x_unit_circle

    center = geom.add_point(*x_center, lc)

    circle_points = []
    for xyz in x_circle:
        p = geom.add_point(*xyz, lc)
        circle_points.append(p)

    circle_arcs = []
    for i in range(len(circle_points) - 1):
        arc = geom.add_circle_arc(circle_points[i], center, circle_points[i + 1])
        circle_arcs.append(arc)
    arc = geom.add_circle_arc(circle_points[-1], center, circle_points[0])
    circle_arcs.append(arc)
    circle_loop = geom.add_curve_loop(circle_arcs)

    entity_dim = 2
    plane_surface_counter = 1
    if tag_counter is not None:
        plane_surface_counter += tag_counter[entity_dim]
    circle_surface = geom.add_plane_surface([circle_loop], tag=plane_surface_counter)
    plane_surface_counter += 1

    # add the rectangle defined by 8 points
    dx = np.array([width / 2, 0.0, 0.0])
    dy = np.array([0.0, height / 2, 0.0])
    x_rectangle = np.stack(
        [
            x_center + dx,
            x_center + dx + dy,
            x_center + dy,
            x_center - dx + dy,
            x_center - dx,
            x_center - dx - dy,
            x_center - dy,
            x_center + dx - dy,
        ]
    )

    rectangle_points = []
    for xyz in x_rectangle:
        p = geom.add_point(*xyz, lc)
        rectangle_points.append(p)

    # draw rectangle lines
    rectangle_lines = []
    for i in range(len(rectangle_points) - 1):
        line = geom.add_line(rectangle_points[i], rectangle_points[i + 1])
        rectangle_lines.append(line)
    line = geom.add_line(rectangle_points[-1], rectangle_points[0])
    rectangle_lines.append(line)

    # connect rectangle points and circle points from outer to inner
    conn = []
    for i in range(len(circle_points)):
        line = geom.add_line(rectangle_points[i], circle_points[i])
        conn.append(line)

    # add curve loops defining surfaces of the matrix
    mat_loops = []
    for i in range(len(circle_points) - 1):
        cloop = geom.add_curve_loop(
            [rectangle_lines[i], conn[i + 1], -circle_arcs[i], -conn[i]]
        )
        mat_loops.append(cloop)
    cloop = geom.add_curve_loop(
        [rectangle_lines[-1], conn[0], -circle_arcs[-1], -conn[-1]]
    )
    mat_loops.append(cloop)

    matrix = []
    for curve_loop in mat_loops:
        mat_surface = geom.add_plane_surface([curve_loop], tag=plane_surface_counter)
        matrix.append(mat_surface)
        plane_surface_counter += 1

    if tag_counter is not None:
        tag_counter[entity_dim] = plane_surface_counter - 1

    if num_cells is not None:
        if not num_cells % 2 == 0:
            raise ValueError(
                "Number of cells per edge must be even for transfinite mesh. Sorry!"
            )

        N = int(num_cells) // 2  # num_cells_per_segment

        for line in circle_arcs:
            geom.mesh.set_transfinite_curve(line, N + 1)
        for line in rectangle_lines:
            geom.mesh.set_transfinite_curve(line, N + 1)
        # diagonal connections
        for line in conn[0::2]:
            geom.mesh.set_transfinite_curve(
                line, N + 3, meshType="Progression", coef=1.0
            )
        # horizontal or vertical connections
        for line in conn[1::2]:
            geom.mesh.set_transfinite_curve(
                line, N + 3, meshType="Progression", coef=0.9
            )

        # transfinite surfaces (circle and matrix)
        # geom.mesh.set_transfinite_surface(tag, arrangement='Left', cornerTags=[])
        geom.mesh.set_transfinite_surface(
            circle_surface, arrangement="AlternateLeft", cornerTags=circle_points[0::2]
        )
        for surface in matrix[0::2]:
            geom.mesh.set_transfinite_surface(surface, arrangement="Right")
        for surface in matrix[1::2]:
            geom.mesh.set_transfinite_surface(surface, arrangement="Left")

        if recombine:
            gmsh.model.geo.mesh.setRecombine(2, circle_surface)
            for surface in matrix:
                gmsh.model.geo.mesh.setRecombine(2, surface)

    geom.synchronize()
    geom.removeAllDuplicates()

    # ### physical groups
    if cell_tags is not None:
        for name, tag in cell_tags.items():
            if name.startswith("matrix"):
                gmsh.model.add_physical_group(2, matrix, tag, name=name)
            elif name.startswith("inclusion"):
                gmsh.model.add_physical_group(2, [circle_surface], tag, name=name)
            else:
                raise NotImplementedError

    # markers for the facets following ordering of
    # entities of multi.dofmap.QuadrilateralDofLayout
    # bottom: 1, left: 2, right: 3, top: 4
    if facet_tags is not None:
        gmsh.model.add_physical_group(1, rectangle_lines[5:7], facet_tags["bottom"], name="bottom")
        gmsh.model.add_physical_group(1, rectangle_lines[3:5], facet_tags["left"], name="left")
        gmsh.model.add_physical_group(
            1, [rectangle_lines[0], rectangle_lines[-1]], facet_tags["right"], name="right"
        )
        gmsh.model.add_physical_group(1, rectangle_lines[1:3], facet_tags["top"], name="top")

    filepath = out_file or "./unit_cell_01.msh"
    _generate_and_write_grid(2, filepath)


def create_unit_cell_02(
    num_cells: Optional[int] = None,
    facets: bool = True,
    out_file: Optional[str] = None,
    options: Optional[dict[str, int]] = None,
):
    """Creates a unit cell with several inclusions.

    Args:
        num_cells: If not None, `num_cells` cells per edge will be created.
        facets: If True, create physical groups for facets.
        out_file: Write mesh to `out_file`. Default: './unit_cell_01.msh'
        options: Options to pass to GMSH. See https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-options
    """

    _initialize()
    _set_gmsh_options(options)
    gmsh.model.add("unit_cell_02")

    xmin = 0.
    xmax = 20.
    ymin = 0.
    ymax = 20.
    z = 0.

    # options
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 2)
    gmsh.option.setNumber("Mesh.Smoothing", 2)

    num_cells = num_cells or 20
    lc_matrix = 20.0 / num_cells
    lc_aggregates = lc_matrix * 0.7

    surfaces_aggregates = []
    curve_loops_aggregates = []
    curve_loop_matrix = []

    def add_aggregate(x, y, z, R):
        """add circle at (x, y, z) with radius R"""
        p1 = gmsh.model.geo.add_point(x, y, z, lc_aggregates)
        p2 = gmsh.model.geo.add_point(x + R, y, z, lc_aggregates)
        p3 = gmsh.model.geo.add_point(x - R, y, z, lc_aggregates)

        c1 = gmsh.model.geo.add_circle_arc(p2, p1, p3)
        c2 = gmsh.model.geo.add_circle_arc(p3, p1, p2)

        loop = gmsh.model.geo.add_curve_loop([c1, c2])
        surface = gmsh.model.geo.add_plane_surface([loop])

        curve_loops_aggregates.append(loop)
        surfaces_aggregates.append(surface)

    bottom = []
    right = []
    top = []
    left = []

    def add_matrix(xmin, xmax, ymin, ymax, z):
        """adds a rectangle from (xmin, ymin, z) to (xmax, ymax, z)"""
        p0 = gmsh.model.geo.add_point(xmin, ymin, z, lc_matrix)
        p1 = gmsh.model.geo.add_point(xmax, ymin, z, lc_matrix)
        p2 = gmsh.model.geo.add_point(xmax, ymax, z, lc_matrix)
        p3 = gmsh.model.geo.add_point(xmin, ymax, z, lc_matrix)
        l0 = gmsh.model.geo.add_line(p0, p1)
        bottom.append(l0)
        l1 = gmsh.model.geo.add_line(p1, p2)
        right.append(l1)
        l2 = gmsh.model.geo.add_line(p2, p3)
        top.append(l2)
        l3 = gmsh.model.geo.add_line(p3, p0)
        left.append(l3)
        loop = gmsh.model.geo.add_curve_loop([l0, l1, l2, l3])

        for line in [l0, l1, l2, l3]:
            gmsh.model.geo.mesh.set_transfinite_curve(line, num_cells + 1)
        curve_loop_matrix.append(loop)

    # add aggregates
    aggregates = [  # (x, y, z, R)
        (8.124435628293494, 16.250990871336494, z, 2.2),
        (3.104265948507514, 3.072789217500327, z, 1.9),
        (16.205618753300654, 16.37885427346391, z, 1.5),
        (3.8648187874608415, 10.576264325380615, z, 2.1),
        (12.807996595076595, 12.686751823841977, z, 1.7),
        (16.23956045449863, 7.686853577410513, z, 1.9),
        (7.9915552082180366, 6.689767983295199, z, 2.0),
        (12.561194629950934, 2.7353694913178512, z, 1.6),
    ]
    for xc, yc, zc, radius in aggregates:
        add_aggregate(xmin + xc, ymin + yc, zc, radius)

    # add the matrix
    add_matrix(xmin, xmax, ymin, ymax, z)
    # add_plane_surface expects list of int (tags of curve loops)
    # if len(arg) > 1 --> subtract curve loops from first
    surface_matrix = gmsh.model.geo.add_plane_surface(
        curve_loop_matrix + curve_loops_aggregates
    )

    # add physical groups
    gmsh.model.geo.synchronize()
    gmsh.model.add_physical_group(2, [surface_matrix], 1, name="matrix")
    gmsh.model.add_physical_group(2, surfaces_aggregates, 2, name="aggregates")

    if facets:
        gmsh.model.add_physical_group(1, bottom, 1, name="bottom")
        gmsh.model.add_physical_group(1, left, 2, name="left")
        gmsh.model.add_physical_group(1, right, 3, name="right")
        gmsh.model.add_physical_group(1, top, 4, name="top")

    filepath = out_file or "./unit_cell_02.msh"
    _generate_and_write_grid(2, filepath)
