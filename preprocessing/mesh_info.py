"""
get info about the mesh parameter for given mesh

Usage:
    mesh_info.py [options] FILE

Arguments:
    FILE           Mesh file incl. extension.

Options:
    -h, --help     Show this message.
"""

import sys
import os
import logging

from pathlib import Path
from docopt import docopt
from dolfin import VectorFunctionSpace, Mesh, XDMFFile


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["FILE"] = Path(args["FILE"])
    return args


def main(args):
    args = parse_arguments(args)
    logger = logging.getLogger("mesh info")

    path = os.path.dirname(os.path.abspath(args["FILE"]))
    base = os.path.splitext(os.path.basename(args["FILE"]))[0]
    ext = os.path.splitext(os.path.basename(args["FILE"]))[1]

    if ext == ".msh":
        import meshio

        mesh = meshio.read(args["FILE"])
        meshio.write(
            args["FILE"].with_suffix(".xdmf"),
            meshio.Mesh(
                points=mesh.points, cells=mesh.cells, cell_data=mesh.cell_data
            )
        )

    mesh = Mesh()
    xdmf = path + "/" + base + ".xdmf"
    with XDMFFile(xdmf) as f:
        f.read(mesh)

    ncells = mesh.num_cells()
    nverts = mesh.coordinates().shape[0]
    hmin = mesh.hmin()
    hmax = mesh.hmax()

    ndofs = [VectorFunctionSpace(mesh, "CG", d).dim() for d in [1, 2]]

    info = f"""Info about {args['FILE']}:
        vertices:       {nverts}
        cells:          {ncells}
        hmin:           {hmin}
        hmax:           {hmax}
        ratio:          {hmax/hmin}\n
        Number of DoFs (VectorFunctionSpace):
            degree 1:   {ndofs[0]}
            degree 2:   {ndofs[1]}\n"""
    logger.warning(info)


if __name__ == "__main__":
    main(sys.argv[1:])
