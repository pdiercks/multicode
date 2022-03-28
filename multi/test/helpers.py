import pygmsh


def build_mesh(NX, NY, LCAR=0.1, order=1, geofile=None, mshfile=None):
    geom = pygmsh.built_in.Geometry()
    geom.add_raw_code("Mesh.SecondOrderIncomplete = 1;")
    XO = 0.0
    YO = 0.0
    X = 1.0 * NX
    Y = 1.0 * NY
    square = geom.add_polygon(
        [[XO, YO, 0.0], [X, YO, 0.0], [X, Y, 0.0], [XO, Y, 0.0]], LCAR
    )

    geom.set_transfinite_surface(square.surface, size=[NX + 1, NY + 1])
    geom.add_raw_code("Recombine Surface {%s};" % square.surface.id)
    geom.add_physical([square.surface], label="square")

    mesh = pygmsh.generate_mesh(
        geom,
        dim=2,
        geo_filename=geofile,
        msh_filename=mshfile,
        prune_z_0=True,
        extra_gmsh_arguments=["-order", f"{order}"],
    )
    return mesh
