from pathlib import Path
import numpy as np
import dolfin as df
import meshio


def read_bases(bases, modes_per_edge=None, return_num_modes=False):
    """read basis functions for multiple reduced bases

    Parameters
    ----------
    bases : list of tuple
        Each element of ``bases`` is a tuple where the first element is a
        FilePath and the second element is a string specifying which basis
        functions to load. Possible string values are 'phi' (coarse
        scale functions), and 'b', 'r', 't', 'l' (for respective fine scale
        functions).
    modes_per_edge : int, optional
        Maximum number of modes per edge for the fine scale bases.
    return_num_modes : bool, optional
        If True, return number of modes per edge.

    Returns
    -------
    B : np.ndarray
        The full reduced basis.
    num_modes : tuple of int
        The maximum number of modes per edge (if ``return_num_modes`` is True).

    """
    loaded = set()
    basis_functions = {}
    num_modes = {}

    # return values
    R = []
    num_max_modes = []

    for filepath, string in bases:
        loaded.add(string)
        npz = np.load(filepath)
        basis_functions[string] = npz[string]
        npz.close()
        num_modes[string] = len(basis_functions[string])

    # check for completeness
    assert not len(loaded.difference(["phi", "b", "r", "t", "l"]))
    loaded.discard("phi")

    max_modes_per_edge = modes_per_edge or max([num_modes[edge] for edge in loaded])

    R.append(basis_functions["phi"])
    for edge in ["b", "r", "t", "l"]:
        rb = basis_functions[edge][:max_modes_per_edge]
        num_max_modes.append(rb.shape[0])
        R.append(rb)

    if return_num_modes:
        return np.vstack(R), tuple(num_max_modes)
    else:
        return np.vstack(R)


class BasesLoader(object):
    def __init__(self, directory, coarse_grid):
        assert directory.is_dir()
        assert coarse_grid._cell.cell_type in ("quad8", "quad9")
        assert hasattr(coarse_grid, "cell_sets")
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

        cells = self.grid.cells
        cell_sets = self.grid.cell_sets

        for cset in cell_sets.values():
            for cell_index, cell in zip(cset, cells[cset]):

                path = self.dir / f"basis_{cell_index:03}.npz"
                self._bases_config[cell_index] = []
                self._bases_config[cell_index].append((path, "phi"))

                self.grid._cell.set_entities(
                    cell
                )  # FIXME not needed if dolfinx.mesh is used
                edges = self.grid._cell.get_entities(
                    dim=1
                )  # TODO implement grid.get_cell_entities(dim) using mesh.topology.connectivity
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
        assert len(self._bases_config.keys()) == self.num_cells


class ResultFile:
    """
    Writes dolfin functions to a XDMF file. Add all functions and products that should be written
    using 'add_function' and 'add_product'. Use 'write' to write the data of the current timestep
    for all added functions and products.
    """

    def __init__(self, filepath, use_local_project=True, **parameters):
        """
        Parameters
        ----------
        filepath
            Path to .xdmf file (incl. extension)
        use_local_project
            Selects the default projection method (True = local - False = global)
        """
        path = Path(filepath).absolute().as_posix()
        assert path.endswith(".xdmf")
        xdmf = df.XDMFFile(path)
        xdmf.parameters["functions_share_mesh"] = parameters.get(
            "functions_share_mesh", True
        )
        xdmf.parameters["rewrite_function_mesh"] = parameters.get(
            "rewrite_function_mesh", False
        )
        xdmf.parameters["flush_output"] = parameters.get("flush_output", True)

        self.xdmf = xdmf
        self.functions = []
        self.product_tuples = []
        self.use_local_project = use_local_project

    def add_function(self, function, name=None):
        """
        Adds a dolfin function to the internal list

        Parameters
        ----------
        function
            Dolfin function that should be written every time 'write' is called
        name : str, optional
            Name of the function.
        """
        if name is not None:
            function.rename(name, function.name())

        self.functions.append(function)

    def add_product(self, product, function_space, name=None, use_local_project=None):
        """
        Adds a dolfin product to the internal list

        Parameters
        ----------
        product
            Dolfin product that should be evaluated and written every time 'Write' is called
        function_space
            Target function space of the projection
        name
            OPTIONAL - New name of the function
        use_local_project
            Overrides the default projection method for this product (True = local - False = global)
        """
        function = df.Function(function_space)
        if name is not None:
            function.rename(name, function.name())

        if use_local_project is None:
            use_local_project = self.use_local_project

        self.product_tuples.append((product, function, use_local_project))

    def write(self, time):
        """
        Writes all stored functions and products to the XDMF-file

        Parameters
        ----------
        time
            Current time.
        """
        for function in self.functions:
            self.xdmf.write(function, time)

        for product_tuple in self.product_tuples:
            product = product_tuple[0]
            function = product_tuple[1]
            use_local_project = product_tuple[2]
            function_space = function.function_space()

            if use_local_project:
                raise NotImplementedError
                # TODO add module multi.projection
                # from multi.projection import local_project

                #  local_project(product, function_space, function)
            else:
                function.assign(df.project(product, function_space))

            self.xdmf.write(
                function,
                function.name(),
                time_step=time,
                encoding=self.file.Encoding,
                append=True,
            )

    def write_checkpoint(self, name, time):
        """Write function (series) to file using df.XDMFFile.write_checkpoint

        Parameters
        ----------
        name : str
            Name of the function to write to file.
        time : float
            Current time.
        """
        append = bool(time)  # assuming t_start = 0.0
        # determine function from name
        function = [f for f in self.functions if f.name() == name][0]
        self.xdmf.write_checkpoint(
            function, name, time, df.XDMFFile.Encoding.HDF5, append
        )

    def read_checkpoint(self, output, name, counter):
        """Read a function using df.XDMFFile.read_checkpoint

        Note
        ----
        This does not work for multiple functions at once.

        Parameters
        ----------
        output : df.function.Function
            A dolfin function to read data into.
        name : str
            Name of the function to read.
        counter: int
            Integer counter used in time-series.
        """
        # determine function from name
        self.xdmf.read_checkpoint(output, name, counter)

    def close(self):
        """close the file"""
        self.xdmf.close()
