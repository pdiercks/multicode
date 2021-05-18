"""
generate a fine grid discretization for a structure of interest (soi) by
duplicating a given RVE grid corresponding to a given COARSE grid.

Usage:
    soi.py [options] COARSE RVE

Arguments:
    COARSE         The coarse grid (incl. msh ext).
    RVE            The RVE grid (incl. msh/xdmf ext).

Options:
    -h, --help                    Show this message.
    -l LEVEL, --log=LEVEL         Set the log level [default: 30].
    -o FILE, --output=FILE        Set output path of final mesh [default: ./soi.xdmf].
    --tdim=TDIM                   The topological dimension of the mesh [default: 2].
    --gdim=GDIM                   The geometrical dimension of the mesh [default: 2].
    --prune_z_0                   Prune zero z-component (when writing to XDMF).
"""

import os
import sys
import tempfile
from pathlib import Path

import meshio
import numpy

import pygmsh
from docopt import docopt
from multi import DofMap
from pymor.core.logger import getLogger


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["COARSE"] = Path(args["COARSE"])
    args["RVE"] = Path(args["RVE"])
    args["--tdim"] = int(args["--tdim"])
    args["--gdim"] = int(args["--gdim"])
    args["--output"] = Path(args["--output"])
    args["--log"] = int(args["--log"])
    return args


def main(args):
    args = parse_arguments(args)
    logger = getLogger("soi")
    logger.setLevel(args["--log"])

    to_be_merged = []
    dofmap = DofMap(
        args["COARSE"],
        tdim=args["--tdim"],
        gdim=args["--gdim"],
    )
    for (cell_index, cell) in enumerate(dofmap.cells):
        logger.debug(f"cell_index: {cell_index}")
        points = numpy.around(dofmap.points[cell], decimals=6)
        x0, y0 = points[0]
        x, y = points[2]

        # compute RVE unit length from mesh
        mesh = meshio.read(args["RVE"])

        # msh2 requires 3D points and 32-bit integers
        if args["RVE"].suffix == ".xdmf":
            # see meshio/gmsh/_gmsh22.py
            if mesh.points.shape[1] == 2:
                mesh.points = numpy.column_stack(
                    [
                        mesh.points[:, 0],
                        mesh.points[:, 1],
                        numpy.zeros(mesh.points.shape[0]),
                    ]
                )

            c_int = numpy.dtype("i")
            for k, (key, value) in enumerate(mesh.cells):
                if value.dtype != c_int:
                    mesh.cells[k] = meshio.CellBlock(
                        key, numpy.array(value, dtype=c_int)
                    )

        # coarse grid cell size needs to agree with RVE unit length (UL)
        UL = abs(numpy.amax(mesh.points[:, 0]) - numpy.amin(mesh.points[:, 0]))
        assert numpy.isclose(x - x0 - UL, 0.0)
        assert numpy.isclose(y - y0 - UL, 0.0)

        # translate the reference RVE
        mesh.points += numpy.array([x0, y0, 0.0])

        # msh2 and msh4 format requires 3D points
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            msh_filename = f.name
            to_be_merged.append(msh_filename)
            meshio.write(msh_filename, mesh, file_format="gmsh22")

    # define geo file which merges several msh files
    geom = pygmsh.built_in.Geometry()
    for msh in to_be_merged:
        geom.add_raw_code(f"Merge '{msh}';")
    geom.add_raw_code("Coherence;")
    geom.add_raw_code("Coherence Mesh;")
    merged_mesh = pygmsh.generate_mesh(
        geom,
        mesh_file_type="msh2",
    )

    has_subdomains = bool(merged_mesh.cell_data_dict)
    assert has_subdomains

    if args["--prune_z_0"]:
        merged_mesh.prune_z_0()

    meshio.write(
        args["--output"],
        meshio.Mesh(
            points=merged_mesh.points,
            cells=merged_mesh.cells,
            cell_data=merged_mesh.cell_data,
        ),
    )

    # clean up
    for msh in to_be_merged:
        os.remove(msh)


if __name__ == "__main__":
    main(sys.argv[1:])
