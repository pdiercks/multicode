import dolfinx
import numpy as np
from mpi4py import MPI
from multi.domain import StructuredQuadGrid


def test():
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, dolfinx.mesh.CellType.quadrilateral)
    grid = StructuredQuadGrid(domain)
    patch_sizes = []
    for i in range(100):
        cell_patch = grid.get_patch(i)
        patch_sizes.append(cell_patch.size)
    assert np.amin(patch_sizes) == 4
    assert np.amax(patch_sizes) == 9
    num_inner = 100 - 4 - 4 * 8
    expected = num_inner * 9 + 4 * 4 + 4 * 8 * 6
    assert np.sum(patch_sizes) == expected

    breakpoint()

"""
1. create mesh with Gmsh and load with gmshio OR use dolfinx to create mesh
2. extract points and cells of the dolfinx.Mesh
3. use points and cells as input to StructuredGrid and DofMap

DofMap requires the mid-points of the second order mesh
StructuredGrid does so as well if to be used with BasesLoader

Question:
    Is the order of the cells the same for mesh.dofmap and V.dofmap?

Parallel:
    Using dolfinx internals (mesh points, cells) in this way probably
    does not work in parallel?
    --> restrict to serial implementation for now ...
"""

# geom_dm = mesh.geometry.dofmap
# points = mesh.geometry.x
# num_cells = geom_dm.offsets.size - 1
# num_nodes = geom_dm.num_nodes
# cells = geom_dm.array.reshape(num_cells, num_nodes)

"""topology

topology only deals with the tags of the entities (vertices, edges, facets)
to get data of actual physical points and their coordinates, one needs to use
mesh.geometry
--> x = mesh.geometry.x # coordinates of all vertices of the mesh
(Note, that a second order mesh (quad9) might have 9 nodes per cell, whereas wrt
the topology the cell (quadrilateral) still only has 4 vertices)
--> mesh.geometry is really how the geometry is interpolated
cmap = mesh.geometry.cmap
cmap.degree (e.g. 1 for 'quad' or 2 for 'quad9' when using Gmsh)

there is also 
layout = cmap.create_dof_layout()
layout.num_entity_dofs(dim)

TODO: It would be nice to be able to define ElementDofLayout,
but currently this is probably to much to handle ..., since I do not know
how the global dofmap is build from this ElementDofLayout


mesh.topology.connectivity(d0, d1)

The adjacency list that for each entity of dimension d0
gives the list of incident entities of dimension d1. Returns
`nullptr` if connectivity has not been computed.

e.g. cell to edges
adj_list = mesh.topology.connectivity(2, 1)
or cell to nodes
adj_list = mesh.topology.connectivity(2, 0)
etc.

Is it possible to re-design multi.DofMap based on dolfinx.mesh ?

ufl_cell = mesh.ufl_cell()
ufl_cell.num_vertices()
ufl_cell.num_edges()
ufl_cell.num_facets()
ufl_cell.facet_types()

NOTE: not all connectivities are already created ...

adjl_cell_to_points = mesh.topology.connectivity(2, 0)
adjl_cell_to_edges = mesh.topology.connectivity(2, 1)
adjl_edge_to_points = mesh.topology.connecitvity(1, 0)

adjl_point_to_cells = mesh.topology.connectivity(0, 2) --> easily query cells that contain given point
adjl_edge_to_cell = mesh.topology.connectivity(1, 2)

mesh.topology.connectivity(1, 1)
mesh.topology.connectivity(2, 2)
both return None (this means they are not created yet)
but these mappings are also useless ...

Question: If I know the vertex tags, how do I get the physical coordinates of those vertices?
--> mesh.geometry.x cannot be used, because the entity tags have no connection to this ...

--> If you want to know the physical coordinates of the mesh you have to interact with
mesh.geometry.dofmap
mesh.geometry.dofmap.links(cell_index) will give vertex tags corresponding to mesh.geometry.x !


Options: 
    (a) pass points and cells to StructuredGrid and leave rest of the impl as is
    (b) pass dolfinx.mesh object and work with the mesh to implement `get_patch` etc.

"""

if __name__ == "__main__":
    test()
