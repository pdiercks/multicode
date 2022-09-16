import gmsh

gmsh.initialize()

gmsh.open("./square.msh")

geom = gmsh.model.geo

model = gmsh.model.get_current()
print(model)
faces = gmsh.model.get_entities(2)

geom.synchronize()

# copy and translate surface of existing model
ss = geom.copy(faces)
breakpoint()
geom.translate(ss, 0.0, 1.0, 0.0)

geom.synchronize()

gmsh.model.mesh.generate(2)
gmsh.write("./vier.msh")
gmsh.finalize()

"""problem

gmsh.merge() is easier
but I cannot translate ...

previous workaround:
    meshio.read
    translate using numpy
    meshio.write
    gmsh.merge

I thought a cleaner solution would be
gmsh.open(...)
but copying and translating the entities seems complicated

"""
