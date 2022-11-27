import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
import yaml

from multi.problems import LinearElasticityProblem
from multi.domain import RectangularDomain
from multi.bcs import BoundaryDataFactory
from multi.extension import extend

# from multi.misc import locate_dofs
# from multi.product import InnerProduct
from multi.shapes import NumpyQuad  # , get_hierarchical_shape_functions

# from scipy.sparse.linalg import eigsh, LinearOperator
# from scipy.special import erfinv

# from pymor.algorithms.gram_schmidt import gram_schmidt
# from pymor.bindings.fenics import FenicsVectorSpace
# from pymor.operators.interface import Operator
# from pymor.vectorarrays.interface import VectorArray


# def construct_hierarchical_basis(
#     problem,
#     max_degree,
#     solver_options=None,
#     orthonormalize=False,
#     product=None,
#     return_edge_basis=False,
# ):
#     """construct hierarchical basis (full space)

#     Parameters
#     ----------
#     problem : multi.problems.LinearProblemBase
#         A suitable problem for which to compute hierarchical
#         edge basis functions.
#     max_degree : int
#         The maximum polynomial degree of the shape functions.
#         Must be greater than or equal to 2.
#     solver_options : dict, optional
#         Solver options in pymor format.
#     orthonormalize : bool, optional
#         If True, orthonormalize the edge basis to inner ``product``.
#     product : optional
#         Inner product wrt to which the edge basis is orthonormalized
#         if ``orthonormalize`` is True.

#     Returns
#     -------
#     basis : VectorArray
#         The hierarchical basis extended into the interior of
#         the domain of the problem.
#     edge_basis : VectorArray
#         The hierarchical edge basis (if ``return_edge_basis`` is True).

#     """
#     V = problem.V
#     try:
#         edge_spaces = problem.edge_spaces
#     except AttributeError as err:
#         raise err("There are no edge spaces defined for given problem.")

#     # ### construct the edge basis on the bottom edge
#     ufl_element = V.ufl_element()
#     L = edge_spaces[0]
#     x_dofs = L.sub(0).collapse().tabulate_dof_coordinates()
#     edge_basis = get_hierarchical_shape_functions(
#         x_dofs[:, 0], max_degree, ncomp=ufl_element.value_size()
#     )
#     source = FenicsVectorSpace(L)
#     B = source.from_numpy(edge_basis)

#     # ### build inner product for edge space
#     product_bc = df.DirichletBC(L, df.Function(L), df.DomainBoundary())
#     inner_product = InnerProduct(L, product, bcs=(product_bc,))
#     product = inner_product.assemble_operator()

#     if orthonormalize:
#         gram_schmidt(B, product=product, copy=False)

#     # ### initialize boundary data
#     basis_length = len(B)
#     Vdim = V.dim()
#     boundary_data = np.zeros((basis_length * len(edge_spaces), Vdim))

#     def mask(index):
#         start = index * basis_length
#         end = (index + 1) * basis_length
#         return np.s_[start:end]

#     # ### fill in values for boundary data
#     for i in range(len(edge_spaces)):
#         boundary_data[mask(i), problem.V_to_L[i]] = B.to_numpy()

#     # ### extend edge basis into the interior of the domain
#     basis = extend_pymor(problem, boundary_data, solver_options=solver_options)
#     if return_edge_basis:
#         return basis, B
#     else:
#         return basis


def compute_phi(problem, nodes):
    """compute coarse scale basis functions for given problem"""
    V = problem.V
    quadrilateral = NumpyQuad(nodes)
    shape_functions = quadrilateral.interpolate(V)

    data_factory = BoundaryDataFactory(problem.domain.grid, V)

    boundary_data = []
    g = dolfinx.fem.Function(V)
    for shape in shape_functions:
        g.x.array[:] = shape
        boundary_data.append([data_factory.create_bc(g)])

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    phi = extend(problem, boundary_data, petsc_options=petsc_options)
    return phi


def compute_coarse_scale_basis(rce_grid, material, degree, out_file):
    """compute the coarse scale basis (extension of bilinear shape functions)

    NOTE
    ----
    method to be used within python action of a dodoFile

    Parameters
    ----------
    rce_grid : filepath
        The partition of the subdomain.
    material : filepath
        The material parameters (.yaml).
    degree : int
        Degree of the VectorFunctionSpace
    """
    domain, cell_marker, facet_marker = gmshio.read_from_msh(
        rce_grid, MPI.COMM_WORLD, gdim=2
    )
    omega = RectangularDomain(domain, cell_marker, facet_marker, index=0)
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", degree))

    # FIXME define nodes via multi.dofmap.QuadrilateralDofLayout ?
    xmin = omega.xmin
    xmax = omega.xmax
    nodes = np.array([
        [xmin[0], xmin[1], 0.],
        [xmax[0], xmin[1], 0.],
        [xmin[0], xmax[1], 0.],
        [xmax[0], xmax[1], 0.]
        ])

    with material.open("r") as f:
        mat = yaml.safe_load(f)

    E = mat["Material parameters"]["E"]["value"]
    NU = mat["Material parameters"]["NU"]["value"]
    plane_stress = mat["Constraints"]["plane_stress"]
    problem = LinearElasticityProblem(omega, V, E=E, NU=NU, plane_stress=plane_stress)
    basis_vectors = compute_phi(problem, nodes)
    out = []
    for vec in basis_vectors:
        out.append(vec.array)

    np.savez(out_file, phi=out)
