import pathlib
import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI

test = pathlib.Path(__file__).parent
msh_file = test / "data" / "block.msh"
mesh, cell_markers, facet_markers = gmshio.read_from_msh(
    msh_file.as_posix(), MPI.COMM_WORLD, gdim=2
)

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

geom_dm = mesh.geometry.dofmap
points = mesh.geometry.x
num_cells = geom_dm.offsets.size - 1
num_nodes = geom_dm.num_nodes
cells = geom_dm.array.reshape(num_cells, num_nodes)

"""topology

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

"""

V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))

# from IPython import embed; embed()
