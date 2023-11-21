from mpi4py import MPI
from dolfinx.io.utils import XDMFFile
import gmsh
import tempfile
import meshio
from multi.preprocessing import create_mesh,  create_rce_grid_01


def merge(mshfiles, output):
    """loads several .msh files and merges them"""
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("merged")

    for msh_file in mshfiles:
        gmsh.merge(msh_file)

    gmsh.model.geo.remove_all_duplicates()
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.remove_duplicate_nodes()
    gmsh.model.mesh.remove_duplicate_elements()

    gmsh.model.mesh.generate(2)

    gmsh.write(output)
    gmsh.finalize()


if __name__ == "__main__":
    coord = [
            [0., 1., 0., 1.],
            [1., 2., 0., 1.]
            ]
    num_cells = 2
    to_be_merged = []
    tfiles = []
    tfiles.append(tempfile.NamedTemporaryFile(suffix=".msh"))
    tfiles.append(tempfile.NamedTemporaryFile(suffix=".msh"))
    for i in range(num_cells):
        tmp = tfiles[i]
        xmin, xmax, ymin, ymax = coord[i]
        create_rce_grid_01(xmin, xmax, ymin, ymax, 
                num_cells=10, facets=False, out_file=tmp.name)
        to_be_merged.append(tmp.name)
    merge(to_be_merged, "final_mesh.msh")
    # cannot be read with dolfin because of same tags for different surfaces etc.

    for tf in tfiles:
        tf.close()

    # write to xdmf using meshio, then dolfinx.io.XDMFFile ...
    in_mesh = meshio.read("final_mesh.msh")

    triangle_mesh = create_mesh(in_mesh, "triangle", prune_z=True)
    meshio.write("final_mesh.xdmf", triangle_mesh)

    with XDMFFile(MPI.COMM_WORLD, "final_mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")

    assert ct.find(0).size == 0
    assert ct.find(1).size > 0
    assert ct.find(2).size > 0
    assert ct.find(3).size == 0
