"""standard and hierarchical shape functions"""
import numpy as np
import sympy
from math import factorial


class NumpyLine:
    tdim = 1

    def __init__(self, nodes, degree):
        assert isinstance(nodes, np.ndarray)
        assert nodes.ndim == self.tdim
        self.nodes = nodes
        self.degree = int(degree)
        # FIXME degree is dependend on nodes
        if self.degree == 1:
            self.G = np.column_stack((np.ones(nodes.size), nodes))
        elif self.degree == 2:
            self.G = np.column_stack((np.ones(nodes.size), nodes, nodes ** 2))
        else:
            raise NotImplementedError

    def interpolate(self, coordinates, gdim):
        """interpolate line shape functions to domain given by coordinates

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
        gdim = int(gdim)
        coordinates = coordinates[:, :gdim]
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


class NumpyQuad4:
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


class NumpyQuad8:
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


def get_hierarchical_shape_1d(p):
    """computes hierarchical polynomial in reference coordinate
    defined as integrand of Legendre polynomials

    Parameters
    ----------
    p : int
        The degree of the Legendre polynomial

    Returns
    -------
    N_p+1 : function
        A polynomial of degree p + 1
    """
    x = sympy.symbols("x")
    if p < 1:
        raise ValueError("p needs to be greater than or equal to 1.")
    N = sympy.diff((x ** 2 - 1) ** p, x, p - 1) / factorial(p - 1) / 2 ** (p - 1)
    return sympy.lambdify(x, N, "numpy")


def mapping(x, a, b, a_tol=1e-3):
    """compute linear mapping from physical (x) to reference
    coordinate (xi)

    map x in [a, b] to xi in [-1, 1]
    xi = alpha * x + beta

    conditions
    (i)  xi(x=a) = -1,
    (ii) xi(x=b) = 1,
    lead to
    beta = (1 + b/a) / (1 - b/a)
    alpha = - 1/a - beta/a

    Parameters
    ----------
    x : np.ndarray
        The physical coordinates.
    a : float
        The lower limit of x.
    b : float
        The upper limit of x.
    a_tol : float, optional
        Use sympy.limit to compute alpha and beta if
        a is smaller than or equal to this value.

    Returns
    -------
    xi : np.ndarray
        The reference coordinates.
    """
    assert b > a
    if a > a_tol:
        beta = (1 + b / a) / (1 - b / a)
        alpha = -1 / a - beta / a
    else:
        # a is zero
        a = sympy.symbols("a")
        f_beta = (1 + b / a) / (1 - b / a)
        f_alpha = -1 / a - f_beta / a
        beta = sympy.limit(f_beta, a, 0)
        alpha = sympy.limit(f_alpha, a, 0)
    return alpha * x + beta


def get_hierarchical_shapes_2d(V, degree):
    """compute hierarchical shapes in V

    Parameters
    ----------
    V
        dolfin FE space.
    degree
        Maximum polynomial degree.

    """
    if degree > 3:
        raise NotImplementedError

    nsub = V.num_sub_spaces()
    if nsub > 0:
        V = V.sub(0).collapse()

    x_dofs = V.tabulate_dof_coordinates()
    x = x_dofs[:, 0]
    y = x_dofs[:, 1]
    xmin = np.amin(x)
    xmax = np.amax(x)
    ymin = np.amin(y)
    ymax = np.amax(y)

    xi = mapping(x, xmin, xmax)
    eta = mapping(y, ymin, ymax)

    # FIXME follow DofMap._cell.enitiy_dofs ...
    """
    gmsh local dof order (up to cubic {8, 9, 10, 11})

    3---6,10---2
    |          |
    7,11       5,9
    |          |
    0---4,8----1

    """
    node_order = {
        0: (0, 0),
        1: (1, 0),
        2: (1, 1),
        3: (0, 1),
    }
    i = 4

    h = {}
    h[0] = lambda x: (1 - x) / 2
    h[1] = lambda x: (1 + x) / 2
    for deg in range(1, degree):
        h[deg + 1] = get_hierarchical_shape_1d(deg)
        higher_degree_nodes = {
            i: (deg + 1, 0),
            i + 1: (1, deg + 1),
            i + 2: (deg + 1, 1),
            i + 3: (0, deg + 1),
        }
        node_order.update(higher_degree_nodes)
        i = (deg + 1) * 4

    H = np.zeros((len(node_order.keys()), V.dim()))
    N_xi = [N(xi) for N in h.values()]
    N_eta = [N(eta) for N in h.values()]
    for i, (p, q) in node_order.items():
        H[i] = N_xi[p] * N_eta[q]

    if nsub:
        return np.kron(H, np.eye(nsub))
    else:
        return H
