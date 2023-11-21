from mpi4py import MPI
import tempfile
import gmsh
import dolfinx
from dolfinx.io import gmshio


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
    gmsh.model.mesh.generate(2)
    gmsh.write(filepath)
    gmsh.finalize()


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(
            0.0, 1.0, 0.0, 1.0, num_cells=(2, 1), recombine=True, out_file=tf.name
        )
        gmsh_grid, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)


    df_grid = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, 2, 1, dolfinx.mesh.CellType.quadrilateral
            )


    def do_test(grid):
        verts = grid.topology.connectivity(2, 0).links(0)
        print(f"{verts=}")
        x = dolfinx.mesh.compute_midpoints(grid, 0, verts)
        print(f"{x=}")
        print(f"{grid.geometry.x=}")


    do_test(gmsh_grid)
    do_test(df_grid)
