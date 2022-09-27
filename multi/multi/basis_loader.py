class BasesLoader(object):
    def __init__(self, directory, dofmap):
        assert directory.is_dir()
        self.dir = directory
        self.dofmap = dofmap
        self.num_cells = dofmap.cells.shape[0]

    def read_bases(self):
        """read basis and max number of modes for each cell in the coarse grid"""
        self._build_bases_config()

        bases = []
        num_max_modes = []
        for cell_index in range(self.num_cells):
            basis, modes = read_bases(self._bases_config[cell_index], return_num_modes=True)
            bases.append(basis)
            num_max_modes.append(modes)

        max_modes = np.vstack(num_max_modes)

        return bases, max_modes

    def _build_bases_config(self):
        """builds logic to read (edge) bases such that a conforming global approx results"""

        marked_edges = []
        self._bases_config = []
        int_to_edge_str = ["b", "r", "t", "l"]

        for cell_index, cell in enumerate(self.dofmap.cells):

            path = self.dir / f"basis_{cell_index:03}.npz"

            self.dofmap._cell.set_entities(cell)
            edges = self.dofmap._cell.get_entities(dim=1)
            for local_ent, ent in enumerate(edges):
                if ent not in marked_edges:
                    marked_edges.append(ent)

                    edge = int_to_edge_str[local_ent]

                    0: ((bases_path[0], ("phi", "b", "r", "t", "l")),),

