"""Design

1. Read mesh created with Gmsh.
2. Instantiate coarse grid.
3. Use class StructuredGrid to define:
    (a) cell sets (problem specific)
4. The info is then used by the BasesLoader to correctly read bases ...

import meshio

mesh = meshio.read(msh_file)
points = mesh.points
cells = mesh.get_cells_type('quad8')
# if physical groups are defined for the boundary
boundary = mesh.get_cells_type('line3')

grid = StructuredGrid(points, cells, tdim)
boundary_cells = grid.get_cells_by_points(boundary)
inner_cells = np.setdiff1d(np.arange(len(cells)), boundary_cells)

# determine 'corner set' for specific problem
x_corner = ...
corner_points = grid.point_at(...)
corner_points = np.append(corner_points, grid.point_at(...))
...
corner_cells = grid.get_cells_by_points(corner_points)
boundary_cells = np.setdiff1d(boundary_cells, corner_cells)

grid.inner_cells = inner_cells
grid.boundary_cells = boundary_cells
grid.corner_cells = corner_cells

loader = BasesLoader(directory, grid)

"""
import numpy as np
from multi.misc import read_bases


class BasesLoader(object):
    def __init__(self, directory, coarse_grid):
        assert directory.is_dir()
        self.dir = directory
        self.grid = coarse_grid
        self.num_cells = coarse_grid.cells.shape[0]

    def read_bases(self):
        """read basis and max number of modes for each cell in the coarse grid"""
        self._build_bases_config()

        bases = []
        num_max_modes = []
        for cell_index in range(self.num_cells):
            basis, modes = read_bases(
                self._bases_config[cell_index], return_num_modes=True
            )
            bases.append(basis)
            num_max_modes.append(modes)

        max_modes = np.vstack(num_max_modes)

        return bases, max_modes

    def _build_bases_config(self):
        """builds logic to read (edge) bases such that a conforming global approx results"""

        marked_edges = {}
        self._bases_config = {}
        int_to_edge_str = ["b", "r", "t", "l"]

        # desired output:
        # bases config for each cell (list of tuple containing npz-file and npz-key)

        # marked_edges is only used to keep track of already marked edges and which file was used
        # because the local_ent is always the one from the current cell

        for cell_index, cell in enumerate(self.grid.inner_cells):

            path = self.dir / f"basis_{cell_index:03}.npz"
            self._bases_config[cell_index] = []
            self._bases_config[cell_index].append((path, "phi"))

            # FIXME self.dofmap does not exist anymore
            # TODO add self._cell to StructuredGrid?
            self.dofmap._cell.set_entities(cell)
            edges = self.dofmap._cell.get_entities(dim=1)
            for local_ent, ent in enumerate(edges):
                edge = int_to_edge_str[local_ent]
                if ent not in marked_edges.keys():
                    # edge is 'visited' the first time
                    # add edge to be loaded from current path
                    self._bases_config[cell_index].append((path, edge))

                    # mark edge as 'visited'
                    marked_edges[ent] = path
                else:
                    # edge was already visited
                    # retrieve `path` from previously marked edges
                    self._bases_config[cell_index].append((marked_edges[ent], edge))

        # TODO
        # same loop for boundary_cells and corner_cells
