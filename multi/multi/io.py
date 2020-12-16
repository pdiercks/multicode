import dolfin as df
from pathlib import Path


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
