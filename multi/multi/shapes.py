"""standard and hierarchical shape functions"""
import numpy as np
import sympy
from math import factorial


class NumpyLine:
    """class to implement nodal Lagrange basis for line elements in 2D space"""

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

    def interpolate(self, function_space, sub):
        """interpolate line shape functions to given Lagrange space.

        Parameters
        ----------
        function_space : dolfin.FunctionSpace
            The Lagrange space used.
        sub : int
            Integer specifying if the line is horizontal (0) or vertical (1).

        Returns
        -------
        phi : np.ndarray
            The shape functions.
        """
        assert function_space.ufl_element().family() in ("Lagrange", "CG")
        if function_space.num_sub_spaces() > 0:
            coordinates = function_space.sub(0).collapse().tabulate_dof_coordinates()
        else:
            coordinates = function_space.tabulate_dof_coordinates()
        coordinates = coordinates[:, : self.gdim]
        coordinates = coordinates[:, sub]
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

        value_shape = function_space.ufl_element().value_shape()
        if value_shape == (2,):
            dim = value_shape[0]
            G = np.zeros((phi.shape[0] * dim, phi.shape[1] * dim))
            for i in range(self.nn):
                G[2 * i, 0::2] = phi[i, :]
                G[2 * i + 1, 1::2] = phi[i, :]

            return G
        elif value_shape == ():
            return phi
        else:
            raise NotImplementedError


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
    """class to implement nodal Lagrange basis for quadrilateral elements"""

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

    def interpolate(self, function_space):
        """interpolate lagrange basis in function_space

        Parameters
        ----------
        function_space : dolfin.FunctionSpace
            The Lagrange space used.

        Returns
        -------
        phi : np.ndarray
            The standard shape functions.

        """
        assert function_space.ufl_element().family() in ("Lagrange", "CG")
        if function_space.num_sub_spaces() > 0:
            coordinates = function_space.sub(0).collapse().tabulate_dof_coordinates()
        else:
            coordinates = function_space.tabulate_dof_coordinates()
        coordinates = coordinates[:, : self.gdim]
        X = get_P_matrix(coordinates, self.nn)
        Id = np.eye(self.nn)
        shapes = []
        for i in range(self.nn):
            coeff = np.linalg.solve(self.P, Id[:, i])
            shapes.append(X @ coeff)
        phi = np.vstack(shapes)

        value_shape = function_space.ufl_element().value_shape()
        if value_shape == (2,):
            dim = value_shape[0]
            G = np.zeros((phi.shape[0] * dim, phi.shape[1] * dim))
            for i in range(self.nn):
                G[2 * i, 0::2] = phi[i, :]
                G[2 * i + 1, 1::2] = phi[i, :]

            return G
        elif value_shape == ():
            return phi
        else:
            raise NotImplementedError


def get_hierarchical_shape_functions(x, max_degree, ncomp=2):
    """construct hierarchical shape functions

    Parameters
    ----------
    x : np.ndarray
        The physical coordinates.
        An array of ``ndim`` 1.
    max_degree : int
        The maximum polynomial degree of the shape functions.
        Must be greater than or equal to 2.
    ncomp : int, optional
        The number of components of the field variable.

    Returns
    -------
    shape_functions : np.ndarray
        The hierarchical shape functions.
        ``len(edge_basis)`` equals ``ncomp * (max_degree-1)``.

    """
    # reference coordinates
    xi = mapping(x, x.min(), x.max())

    shapes = []
    for degree in range(2, max_degree + 1):
        fun = _get_hierarchical_shape_fun_expr(degree)
        shapes.append(fun(xi))

    shape_functions = np.kron(shapes, np.eye(ncomp))
    return shape_functions


def _get_hierarchical_shape_fun_expr(degree):
    """get hierarchical shape function of degree

    Note
    ----
    For degree >= 2 return the integrand of the Legendre polynomial of degree
    p = degree - 1.
    The functions are defined on the interval [-1, 1].
    This method implements equation (8.61) in the book
    "The finite element method volume 1" by Zienkiewicz and Taylor

    Parameters
    ----------
    degree : int
        The degree of the hierarchical shape function.

    Returns
    -------
    shape_function : function
        A polynomial of degree `degree`.
    """
    if degree < 2:
        raise NotImplementedError
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
        beta = np.float64(sympy.limit(f_beta, a, 0))
        alpha = np.float64(sympy.limit(f_alpha, a, 0))
    return alpha * x + beta
