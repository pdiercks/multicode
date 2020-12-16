"""
convert gmsh geo file to xdmf readable by dolfin 2019.1.0

Usage:
    geo_to_xdmf.py [options] GEO

Arguments:
    GEO       The geo file to convert.

Options:
    -h, --help               Show this message.
    -o FILE, --output=FILE   Output XDMFFile. If `None` the geo filepath
                             with suffix `.xdmf` is used by default.
    --prune_z_0              Prune z coordinates.
    --summary                Print mesh summary.
"""

import sys
from docopt import docopt
import meshio
from pathlib import Path
from subprocess import call


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["GEO"] = Path(args["GEO"])
    args["--output"] = Path(args["--output"]) if args["--output"] else None
    return args


def main(args):
    args = parse_arguments(args)
    xdmf = create_xdmf(args)
    if args["--summary"]:
        get_N_dofs(xdmf)


def create_xdmf(args):
    geo = args["GEO"]
    if args["--output"]:
        xdmf = args["--output"]
        msh = xdmf.with_suffix(".msh")
    else:
        xdmf = geo.with_suffix(".xdmf")
        msh = geo.with_suffix(".msh")

    call(["gmsh", "-2", "-order", "1", "-format", "msh2", f"{geo}", "-o", f"{msh}"])

    geometry = meshio.read(msh)
    if args["--prune_z_0"]:
        geometry.points = geometry.points[:, :2]

    meshio.write(
        xdmf,
        meshio.Mesh(
            points=geometry.points, cells=geometry.cells, cell_data=geometry.cell_data
        ),
    )
    return xdmf


def get_N_dofs(xdmf_file):
    from dolfin import (
        XDMFFile,
        Mesh,
        VectorFunctionSpace,
        MeshValueCollection,
        MeshFunction,
    )

    mesh = Mesh()
    mvc = MeshValueCollection("size_t", mesh, 1)

    with XDMFFile(xdmf_file) as f:
        f.read(mesh)
        f.read(mvc, "gmsh:physical")

    subdomains = MeshFunction("size_t", mesh, mvc)
    dofs = []
    for degree in [1, 2]:
        V = VectorFunctionSpace(mesh, "CG", degree)
        dofs.append(V.dim())

    print(
        f"""Mesh Summary:
          Vertices:    {mesh.coordinates().shape[0]}
          DoFs:
              degree 1:    {dofs[0]}
              degree 2:    {dofs[1]}\n"""
    )


if __name__ == "__main__":
    main(sys.argv[1:])
