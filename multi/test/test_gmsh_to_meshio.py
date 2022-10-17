import gmsh
from multi.domain import extract_to_meshio

rce_path = "/home/pdiercks/repos/bam/2020_02_multiscale/work/convergence/rce_01/rce_05.msh"

gmsh.initialize()
gmsh.model.add("fine_grid")

gmsh.open(rce_path)

gmsh.model.geo.synchronize()

test = extract_to_meshio()

breakpoint()


gmsh.finalize()
