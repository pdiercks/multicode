"""standard and hierarchical shape functions"""
import numpy as np
import sympy
from math import factorial


class NumpyLine:
    tdim = 1
    gdim = 2

    def __init__(self, nodes):
        assert isinstance(nodes, np.ndarray)
        assert nodes.ndim == self.tdim
        self.nodes = nodes
        self.nn = nodes.shape[0]
        if self.nn == 2:
            self.G = np.column_stack((np.ones(nodes.size), nodes))
        elif self.nn == 3:
            self.G = np.column_stack((np.ones(nodes.size), nodes, nodes ** 2))
        else:
            raise NotImplementedError

    def interpolate(self, coordinates, value_shape):
        """interpolate line shape functions to domain given by coordinates

        Parameters
        ----------
        coordinates : np.ndarray
            The dof coordinates of the SCALAR function space
            (V.tabulate_dof_coordinates()).
        value_shape : tuple of int
            The shape of a function f in V.

        Returns
        -------
        phi : np.ndarray
            The shape functions.
        """
        coordinates = coordinates[:, : self.gdim]
        if self.nn == 2:
            X = np.column_stack(
                (
                    np.ones(coordinates.size),
                    coordinates,
                )
            )
        elif self.nn == 3:
            X = np.column_stack(
                (np.ones(coordinates.size), coordinates, coordinates ** 2)
            )
        else:
            assert False

        Id = np.eye(self.nn)
        shapes = []
        for i in range(self.nn):
            coeff = np.linalg.solve(self.G, Id[:, i])
            shapes.append(X @ coeff)
        phi = np.vstack(shapes)

        if len(value_shape) > 1:
            raise NotImplementedError
        dim = value_shape[0]
        if dim == 2:
            G = np.zeros((phi.shape[0] * dim, phi.shape[1] * dim))
            for i in range(self.nn):
                G[2 * i, 0::2] = phi[i, :]
                G[2 * i + 1, 1::2] = phi[i, :]

            return G
        else:
            return phi


def get_P_matrix(X, nn):
    """get matrix of variables such that

    np.dot(P, coeff) = e_i, where
    coeff are the coefficients of the shape function N
    such that N_i(n_j) = delta_ij
    """
    x = X[:, 0]
    y = X[:, 1]
    variables = [
        np.ones(X.shape[0]),
        x,
        y,
        x * y,
        x ** 2,
        y ** 2,
        x ** 2 * y,
        x * y ** 2,
        x ** 2 * y ** 2,
        x ** 3,
        y ** 3,
        x ** 3 * y,
        x * y ** 3,
        x ** 3 * y ** 2,
        x ** 2 * y ** 3,
        x ** 3 * y ** 3,
    ]
    return np.column_stack(variables[:nn])


class NumpyQuad:
    tdim = 2
    gdim = 2

    def __init__(self, nodes):
        """initialize quadrilateral with nodes

        Parameters
        ----------
        nodes : list or np.ndarray
            Node coordinates of the quadrilateral.
        """
        if isinstance(nodes, list):
            nodes = np.array(nodes).reshape(len(nodes), len(nodes[0]))
        assert isinstance(nodes, np.ndarray)
        # prune zero z component
        self.nodes = nodes[:, : self.gdim]
        self.nn = self.nodes.shape[0]
        self.P = get_P_matrix(self.nodes, self.nn)

    def interpolate(self, coordinates, value_shape):
        """interpolate lagrange basis in FE space V

        Parameters
        ----------
        coordinates : np.ndarray
            The dof coordinates of the space V.
        value_shape: tuple
            The shape of a function in V.

        Returns
        -------
        phi : np.ndarray
            The standard shape functions.

        """
        coordinates = coordinates[:, : self.gdim]
        X = get_P_matrix(coordinates, self.nn)
        Id = np.eye(self.nn)
        shapes = []
        for i in range(self.nn):
            coeff = np.linalg.solve(self.P, Id[:, i])
            shapes.append(X @ coeff)
        phi = np.vstack(shapes)

        if len(value_shape) > 1:
            raise NotImplementedError
        dim = value_shape[0]
        if dim == 2:
            G = np.zeros((phi.shape[0] * dim, phi.shape[1] * dim))
            for i in range(self.nn):
                G[2 * i, 0::2] = phi[i, :]
                G[2 * i + 1, 1::2] = phi[i, :]

            return G
        else:
            return phi


def get_hierarchical_shape_1d(degree):
    """get hierarchical shape function of degree

    Note
    ----
    If degree < 2 linear functions (1 ± x) / 2 are returned.
    For degree >= 2 return the integrand of the Legendre polynomial of degree
    p = degree - 1.

    Parameters
    ----------
    degree : int
        The degree of the hierarchical shape function.

    Returns
    -------
    N : function
        A polynomial of degree if degree >= 2 else (1 ± x) / 2
    """
    if degree == 0:
        return lambda x: (1 - x) / 2
    elif degree == 1:
        return lambda x: (1 + x) / 2
    else:
        p = degree - 1
        x = sympy.symbols("x")
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

    from multi.dofmap import Quadrilateral

    # TODO extend up to any degree?
    if degree > 3:
        raise NotImplementedError
    ndofs_per_edge = degree - 1
    ndofs_per_vert = 1
    cell = Quadrilateral("quad8")
    cell.set_entity_dofs(ndofs_per_vert, ndofs_per_edge, 0)

    #  cell.enitiy_dofs ordering
    """e.g. for degree=3, cubic dofs = (5, 7, 9, 11)

    3----8,9----2
    |           |
    10,11       6,7
    |           |
    0----4,5----1

    """

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

    def get_indices(d):
        if d < 2:
            return [(0, 0), (1, 0), (1, 1), (0, 1)]
        else:
            return [(d, 0), (1, d), (d, 1), (0, d)]

    deg_to_entdim = {0: 0, 1: 0, 2: 1, 3: 1}

    h = {}
    for deg in range(degree + 1):
        h[deg] = get_hierarchical_shape_1d(deg)

    # TODO why so complicated?
    node_indices = {}
    entity_dofs = cell.get_entity_dofs()
    for deg in range(1, degree + 1):
        dim = deg_to_entdim[deg]
        entities = entity_dofs[dim].keys()
        indices = get_indices(deg)
        for entity in entities:
            if deg > 1:
                dofs = entity_dofs[dim][entity][: deg - 1]
            else:
                dofs = entity_dofs[dim][entity]
            for dof in dofs:
                if dof not in node_indices.keys():
                    node_indices.update({dof: indices[entity]})

    H = np.zeros((len(node_indices.keys()), V.dim()))
    N_xi = [N(xi) for N in h.values()]
    N_eta = [N(eta) for N in h.values()]
    for i, (p, q) in node_indices.items():
        H[i] = N_xi[p] * N_eta[q]

    if nsub:
        return np.kron(H, np.eye(nsub))
    else:
        return H
