"""Idea

read model, copy entities, translate
write (duplicated) mesh
"""
import gmsh

gmsh.initialize()

gmsh.model.add("square")

geom = gmsh.model.geo

lc = 1.0
p0 = geom.add_point(0, 0, 0, lc)
p1 = geom.add_point(1, 0, 0, lc)
p2 = geom.add_point(1, 1, 0, lc)
p3 = geom.add_point(0, 1, 0, lc)

l0 = geom.add_line(p0, p1)
l1 = geom.add_line(p1, p2)
l2 = geom.add_line(p2, p3)
l3 = geom.add_line(p3, p0)

ll = geom.add_curve_loop([l0, l1, l2, l3])

s = geom.add_plane_surface([ll])

geom.synchronize()

# copy and translate surface of existing model
ss = geom.copy([(2, s)])
geom.translate(ss, 1.0, 0.0, 0.0)

geom.remove_all_duplicates()
geom.synchronize()

gmsh.model.mesh.generate(2)
gmsh.write("./square.msh")
gmsh.finalize()
