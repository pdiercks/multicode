import gmsh
import meshio

gmsh.initialize()

xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0
z = 0.0
lc = 0.1

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

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(2)
gmsh.write("nogroup.msh")
gmsh.finalize()


m = meshio.read("nogroup.msh")
meshio.write("nogroup_io.msh", m)
# meshio.write("nogroup_io.msh", m, file_format="gmsh")
