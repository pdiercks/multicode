import meshio
import gmsh


gmsh.initialize()
gmsh.model.add("model")

gmsh.merge("./r1.msh")
gmsh.merge("./r2.msh")

gmsh.model.geo.removeAllDuplicates()  # equivalent to Coherence; ?
gmsh.model.mesh.removeDuplicateNodes()  # equivalent to Coherence Mesh; ?

gmsh.model.mesh.generate(2)
gmsh.write("./result.msh")
gmsh.finalize()

m = meshio.read("./result.msh")
