import dolfin as df
import numpy as np
import sympy


def apply_mixed_bcs(matrix, coarse_dofs, fine_dofs):
    """set boundary conditions to matrices with mixed scales"""
    assert isinstance(matrix, np.ndarray)
    cdofs = [] if coarse_dofs is None else list(coarse_dofs)
    fdofs = [] if fine_dofs is None else list(fine_dofs)
    nrow, ncol = matrix.shape

    if nrow < ncol:
        # apply coarse scale bcs
        row = np.zeros(ncol)
        row[fdofs] = 1
        matrix[cdofs, :] = row
    elif nrow > ncol:
        # apply fine scale bcs
        row = np.zeros(ncol)
        row[cdofs] = 1
        matrix[fdofs] = row
    else:
        # assert nrow == ncol
        # apply bcs on same scale matrix
        assert coarse_dofs == fine_dofs
        matrix[cdofs] = np.eye(ncol)[cdofs]
    return matrix


class CoarseElementBase:
    """Base class for coarse grid elements - do not use directly"""

    def __init__(self, coord):
        assert isinstance(coord, np.ndarray)
        self.x, self.y, self.z = sympy.symbols("x[0] x[1] x[2]")
        self.coord = coord
        self.ones = np.ones(coord.shape[0])

    def interpolate(self, space):
        """interpolate shape function in given space and build projection matrix"""
        basis = []
        for N in self.shapes:
            expr = []
            # TODO this seems to be rather slow
            if self.gdim == 1:
                expr.append(df.Expression(sympy.printing.ccode(N), degree=self.degree))
            elif self.gdim == 2:
                expr.append(
                    df.Expression((sympy.printing.ccode(N), 0.0), degree=self.degree)
                )
                expr.append(
                    df.Expression((0.0, sympy.printing.ccode(N)), degree=self.degree)
                )
            else:
                raise NotImplementedError
            for e in expr:
                p = df.interpolate(e, space)
                basis.append(p.vector().get_local())
        return np.array(basis)


class NumpyGlobalLine:
    tdim = 1

    def __init__(self, nodes, degree):
        assert isinstance(nodes, np.ndarray)
        assert nodes.ndim == self.tdim
        self.nodes = nodes
        self.degree = int(degree)
        if self.degree == 1:
            self.G = np.column_stack((np.ones(nodes.size), nodes))
        elif self.degree == 2:
            self.G = np.column_stack((np.ones(nodes.size), nodes, nodes ** 2))
        else:
            raise NotImplementedError

    def interpolate(self, coordinates, gdim):
        gdim = int(gdim)
        assert coordinates.ndim == self.tdim
        assert gdim in (1, 2)
        if self.degree == 1:
            X = np.column_stack((np.ones(coordinates.size), coordinates,))
        elif self.degree == 2:
            X = np.column_stack(
                (np.ones(coordinates.size), coordinates, coordinates ** 2)
            )
        else:
            assert False

        NP = 2 if self.degree == 1 else 3
        I = np.eye(NP)
        shapes = []
        for i in range(NP):
            coeff = np.linalg.solve(self.G, I[:, i])
            shapes.append(X @ coeff)
        phi = np.vstack(shapes)

        if gdim == 2:
            G = np.zeros((phi.shape[0] * gdim, phi.shape[1] * gdim))
            for i in range(NP):
                G[2 * i, 0::2] = phi[i, :]
                G[2 * i + 1, 1::2] = phi[i, :]

            return G
        else:
            return phi


class NumpyGlobalQ4:
    tdim = 2
    degree = 1

    def __init__(self, nodes):
        assert isinstance(nodes, np.ndarray)
        self.nodes = nodes
        self.G = np.column_stack(
            (np.ones(nodes.shape[0]), nodes, nodes[:, 0] * nodes[:, 1])
        )

    def interpolate(self, coordinates, gdim):
        """interpolate 4 node quadrilateral to domain given by coordinates

        Parameters
        ----------
        coordinates : np.ndarray
            The dof coordinates of the SCALAR function space
            (V.tabulate_dof_coordinates()).
        gdim : int
            Geometrical dimension of the function space.

        Returns
        -------
        phi : np.ndarray
            The shape functions.
        """
        assert gdim in (1, 2)
        coordinates = coordinates[:, :gdim]
        X = np.column_stack(
            (
                np.ones(coordinates.shape[0]),
                coordinates,
                coordinates[:, 0] * coordinates[:, 1],
            )
        )
        I = np.eye(4)
        shapes = []
        for i in range(4):
            coeff = np.linalg.solve(self.G, I[:, i])
            shapes.append(X @ coeff)
        phi = np.vstack(shapes)

        if gdim == 2:
            G = np.zeros((phi.shape[0] * gdim, phi.shape[1] * gdim))
            for i in range(4):
                G[2 * i, 0::2] = phi[i, :]
                G[2 * i + 1, 1::2] = phi[i, :]

            return G
        else:
            return phi


class GlobalQ4(CoarseElementBase):
    """global shape function for bilinear quadrilateral element"""

    gdim = 2
    degree = 1

    def __init__(self, coord):
        super().__init__(coord)
        shapes = []
        X = np.array([1, self.x, self.y, self.x * self.y])
        G = np.column_stack(
            (self.ones, self.coord, self.coord[:, 0] * self.coord[:, 1])
        )
        for i in range(4):
            coeff = np.linalg.solve(G, np.eye(4)[:, i])
            shapes.append(np.dot(coeff, X))
        self.shapes = np.array(shapes)


class NumpyGlobalQ8:
    tdim = 2
    degree = 2

    def __init__(self, nodes):
        assert isinstance(nodes, np.ndarray)
        self.nodes = nodes
        self.G = np.column_stack(
            (
                np.ones(nodes.shape[0]),
                self.nodes,
                self.nodes[:, 0] * self.nodes[:, 1],
                self.nodes[:, 0] ** 2,
                self.nodes[:, 1] ** 2,
                self.nodes[:, 0] ** 2 * self.nodes[:, 1],
                self.nodes[:, 0] * self.nodes[:, 1] ** 2,
            )
        )

    def interpolate(self, coordinates, gdim):
        """interpolate 8 node quadrilateral to domain given by coordinates

        Parameters
        ----------
        coordinates : np.ndarray
            The dof coordinates of the SCALAR function space
            (V.tabulate_dof_coordinates()).
        gdim : int
            Geometrical dimension of the function space.

        Returns
        -------
        phi : np.ndarray
            The shape functions.
        """
        assert gdim in (1, 2)
        X = np.column_stack(
            (
                np.ones(coordinates.shape[0]),
                coordinates,
                coordinates[:, 0] * coordinates[:, 1],
                coordinates[:, 0] ** 2,
                coordinates[:, 1] ** 2,
                coordinates[:, 0] ** 2 * coordinates[:, 1],
                coordinates[:, 0] * coordinates[:, 1] ** 2,
            )
        )
        I = np.eye(8)
        shapes = []
        for i in range(8):
            coeff = np.linalg.solve(self.G, I[:, i])
            shapes.append(X @ coeff)
        phi = np.vstack(shapes)
        assert not np.any(np.isnan(phi))

        if gdim == 2:
            G = np.zeros((phi.shape[0] * gdim, phi.shape[1] * gdim), dtype=phi.dtype)
            for i in range(8):
                G[2 * i, 0::2] = phi[i, :]
                G[2 * i + 1, 1::2] = phi[i, :]

            return G
        else:
            return phi


class GlobalQ8(CoarseElementBase):
    """global shape function for quadratic quadrilateral element"""

    gdim = 2
    degree = 2

    def __init__(self, coord):
        super().__init__(coord)
        shapes = []
        X = np.array(
            [
                1,
                self.x,
                self.y,
                self.x * self.y,
                self.x * self.x,
                self.y * self.y,
                self.x * self.x * self.y,
                self.x * self.y * self.y,
            ]
        )
        G = np.column_stack(
            (
                self.ones,
                self.coord,
                self.coord[:, 0] * self.coord[:, 1],
                self.coord[:, 0] ** 2,
                self.coord[:, 1] ** 2,
                self.coord[:, 0] ** 2 * self.coord[:, 1],
                self.coord[:, 0] * self.coord[:, 1] ** 2,
            )
        )
        for i in range(8):
            coeff = np.linalg.solve(G, np.eye(8)[:, i])
            shapes.append(np.dot(coeff, X))
        self.shapes = np.array(shapes)


class GlobalQ9(CoarseElementBase):
    """global shape function for biquadratic quadrilateral element"""

    gdim = 2
    degree = 2

    def __init__(self, coord):
        super().__init__(coord)
        X = np.array(
            [
                [
                    1,
                    self.x,
                    self.y,
                    self.x * self.y,
                    self.x * self.x,
                    self.y * self.y,
                    self.x * self.x * self.y,
                    self.x * self.y * self.y,
                    self.x * self.x * self.y * self.y,
                ],
            ]
        )
        G = np.column_stack(
            (
                self.ones,
                self.coord,
                self.coord[:, 0] * self.coord[:, 1],
                self.coord[:, 0] ** 2,
                self.coord[:, 1] ** 2,
                self.coord[:, 0] ** 2 * self.coord[:, 1],
                self.coord[:, 0] * self.coord[:, 1] ** 2,
                self.coord[:, 0] ** 2 * self.coord[:, 1] ** 2,
            )
        )
        H = sympy.Matrix(np.linalg.inv(G))  # TODO assert G is not singular
        self.shapes = X * H


class GlobalL2(CoarseElementBase):
    """global shape function for linear line element"""

    gdim = 1
    degree = 1

    def __init__(self, coord):
        super().__init__(coord)
        X = np.array([[1, self.x],])
        G = np.column_stack((self.ones, self.coord,))
        H = sympy.Matrix(np.linalg.inv(G))  # TODO assert G is not singular
        self.shapes = X * H


class GlobalL3(CoarseElementBase):
    """global shape function for quadratic line element"""

    gdim = 1
    degree = 2

    def __init__(self, coord):
        super().__init__(coord)
        X = np.array([[1, self.x, self.x * self.x],])
        G = np.column_stack((self.ones, self.coord, self.coord ** 2))
        H = sympy.Matrix(np.linalg.inv(G))  # TODO assert G is not singular
        self.shapes = X * H
