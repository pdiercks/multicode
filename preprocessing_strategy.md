Preprocessing with Gmsh
=======================

Always write the grid in .msh format.

Reasoning:
(a) Meshes can be merged without problem.
(b) Preparing the mesh for dolfin input is not always general (celltype, gdim, meshtags)
but this info is required when writing to .xdmf format.

(b) For simulation:
• read .msh with meshio
• use `multi.preprocessing.create_mesh` (celltype info needed; might depend on application)
