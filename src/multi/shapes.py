"""standard and hierarchical shape functions"""

from typing import Callable
import numpy as np
import numpy.typing as npt
import sympy
from math import factorial
from dolfinx import fem


class NumpyLine:
    """class to implement nodal Lagrange basis for line elements in 2D space"""

    tdim = 1
    gdim = 2

    def __init__(self, nodes: npt.NDArray):
        assert nodes.ndim == self.tdim
        self.nodes = nodes
        self.nn = nodes.shape[0]
        if self.nn == 2:
            self.G = np.column_stack((np.ones(nodes.size), nodes))
        elif self.nn == 3:
            self.G = np.column_stack((np.ones(nodes.size), nodes, nodes**2))
        else:
            raise NotImplementedError

    def interpolate(self, function_space: fem.FunctionSpaceBase, orientation: int) -> npt.NDArray:
        """interpolate line shape functions to given Lagrange space.

        Args:
            function_space: The FE space.
            orientation: Integer specifying if the line is horizontal (0) or vertical (1).

        """
        assert function_space.ufl_element().family() in ("Lagrange", "P")
        assert isinstance(orientation, int)
        assert 0 <= orientation <= 1
        coordinates = function_space.tabulate_dof_coordinates()
        coordinates = coordinates[:, : self.gdim]
        coordinates = coordinates[:, orientation]
        if self.nn == 2:
            X = np.column_stack((np.ones(coordinates.size), coordinates))
        else:
            X = np.column_stack(
                (np.ones(coordinates.size), coordinates, coordinates**2)
            )
        Id = np.eye(self.nn)
        shapes = []
        for i in range(self.nn):
            coeff = np.linalg.solve(self.G, Id[:, i])
            shapes.append(X @ coeff)
        phi = np.vstack(shapes) # shape (num_nodes, Vdim)

        value_shape = function_space.ufl_element().value_shape()
        if len(value_shape) > 0:
            # vector valued function space
            ncomp = value_shape[0] # either 2 or 3
            G = np.zeros((phi.shape[0] * ncomp, phi.shape[1] * ncomp))
            for i in range(self.nn):
                for j in range(ncomp):
                    G[ncomp * i + j, j::ncomp] = phi[i, :]
            return G
        else:
            # scalar function space
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
        x**2,
        y**2,
        x**2 * y,
        x * y**2,
        x**2 * y**2,
        x**3,
        y**3,
        x**3 * y,
        x * y**3,
        x**3 * y**2,
        x**2 * y**3,
        x**3 * y**3,
    ]
    return np.column_stack(variables[:nn])


class NumpyQuad:
    """class to implement nodal Lagrange basis for quadrilateral elements"""

    tdim = 2
    gdim = 2

    def __init__(self, nodes: npt.NDArray):
        """initialize quadrilateral with nodes

        Args:
            nodes: Node coordinates of the quadrilateral.
        """
        assert isinstance(nodes, np.ndarray)
        assert nodes.shape[-1] <= 3
        # shape = (num_nodes, gdim)
        # prune zero z component
        self.nodes = nodes[:, : self.gdim]
        self.nn = nodes.shape[0]
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
        assert function_space.ufl_element().family() in ("P", "Q", "Lagrange")
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
        if len(value_shape) > 0:
            # vector valued function space
            ncomp = value_shape[0] # either 2 or 3
            G = np.zeros((phi.shape[0] * ncomp, phi.shape[1] * ncomp))
            for i in range(self.nn):
                for j in range(ncomp):
                    G[ncomp * i + j, j::ncomp] = phi[i, :]
            return G
        else:
            # scalar function space
            return phi

def get_hierarchical_shape_functions(x: npt.NDArray, max_degree: int, ncomp: int = 2) -> npt.NDArray:
    """Constructs hierarchical shape functions.

    Args:
        x: The physical coordinates of the function space (interval mesh).
        max_degree: Max polynomial degree. Must be >= 2.
        ncomp: Number of components of the field variable.

    Returns:
        shape_functions : The hierarchical shape functions.

    """
    if max_degree < 2:
        raise ValueError("Requires max_degree >= 2.")
    # reference coordinates
    xi = mapping(x, x.min(), x.max())

    shapes = []
    for degree in range(2, max_degree + 1):
        fun = _get_hierarchical_shape_fun_expr(degree)
        shapes.append(fun(xi))

    shape_functions = np.kron(np.array(shapes), np.eye(ncomp))
    return shape_functions


def _get_hierarchical_shape_fun_expr(degree: int) -> Callable:
    """Returs hierarchical shape function expression as 'lambdified' callable.

    Args:
        degree: The degree of the hierarchical shape function.

    Notes:
        For degree >= 2 return the integrand of the Legendre polynomial of degree p = `degree` - 1.
        The functions are defined on the interval [-1, 1].
        This method implements equation (8.61) in the book
        "The finite element method volume 1" by Zienkiewicz and Taylor

    Returns:
        shape_function: A hierarchical shape function of degree `degree`.

    """
    p = degree - 1
    x = sympy.symbols("x")
    N = sympy.diff((x**2 - 1) ** p, x, p - 1) / factorial(p - 1) / 2 ** (p - 1) # type: ignore
    return sympy.lambdify(x, N, "numpy")


def mapping(x: npt.NDArray, a: float, b: float, a_tol: float = 1e-3) -> npt.NDArray:
    """Computes linear mapping from physical (x) to reference coordinate (xi).

    Args:
        x: The phyiscal coordinates.
        a: Min value of x.
        b: Max value of x.
        a_tol: If a <= a_tol, assume a=0.

    Returns:
        xi: The reference coordinates.

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
