import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
import yaml

from multi.problems import LinearElasticityProblem
from multi.domain import RceDomain
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


def compute_phi(problem):
    """compute coarse scale basis functions for given problem"""
    V = problem.V
    # FIXME just define nodes based on domain.geometry?
    # get_corner_vertices might be a bit too much effort
    vertices = problem.domain.get_corner_vertices()
    nodes = dolfinx.mesh.compute_midpoints(problem.domain.mesh, 0, vertices)
    quadrilateral = NumpyQuad(nodes)
    shape_functions = quadrilateral.interpolate(V)

    data_factory = BoundaryDataFactory(problem.domain.mesh, V)
    bc_dofs = data_factory.boundary_dofs

    boundary_data = []
    for shape in shape_functions:
        f = data_factory.create_function(shape[bc_dofs], bc_dofs)
        boundary_data.append([data_factory.create_bc(f)])

    phi = extend(problem, boundary_data)
    return phi


def compute_coarse_scale_basis(rce_grid, material, degree, out_file):
    """method to be used within python action of a dodoFile

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
    omega = RceDomain(domain, cell_marker, facet_marker, index=0, edges=True)
    V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", degree))

    with material.open("r") as f:
        mat = yaml.safe_load(f)

    E = mat["Material parameters"]["E"]["value"]
    NU = mat["Material parameters"]["NU"]["value"]
    plane_stress = mat["Constraints"]["plane_stress"]
    problem = LinearElasticityProblem(omega, V, E=E, NU=NU, plane_stress=plane_stress)
    basis_vectors = compute_phi(problem)
    out = []
    for vec in basis_vectors:
        out.append(vec.array)

    np.savez(out_file, phi=out)


# adapted from pymor.algorithms.randrangefinder.adaptive_rrf
# to be used with multi.oversampling.OversamplingProblem
# def compute_fine_scale_snapshots(
#     logger,
#     oversampling_problem,
#     coarse_scale_basis,
#     source_product=None,
#     range_product=None,
#     tol=1e-4,
#     failure_tolerance=1e-15,
#     num_testvecs=20,
#     lambda_min=None,
#     train_vectors=None,
# ):
#     r"""Adaptive randomized range approximation of `A`.
#     This is an implementation of Algorithm 1 in [BS18]_.

#     The following modifications where made:
#         - instead of a transfer operator A, the full oversampling
#         problem is used
#         - optional training data was added (`train_vectors`)

#     The image Av = A.apply(v) is equivalent to the restriction
#     of the full solution to the target domain Ω_in, i.e.
#         U = oversampling_problem.solve(v)
#     See multi.oversampling.OversamplingProblem.solve.

#     Given the |Operator| `A`, the return value of this method is the |VectorArray|
#     `B` with the property

#     .. math::
#         \Vert A - P_{span(B)} A \Vert \leq tol

#     with a failure probability smaller than `failure_tolerance`, where the norm denotes the
#     operator norm. The inner product of the range of `A` is given by `range_product` and
#     the inner product of the source of `A` is given by `source_product`.

#     Parameters
#     ----------
#     logger
#         The logger used by the main program.
#     oversampling_problem
#         The full oversampling problem associated with a (transfer) |Operator| A.
#     coarse_scale_basis
#         The |VectorArray| containing the coarse scale basis.
#     source_product
#         Inner product |Operator| of the source of A.
#     range_product
#         Inner product |Operator| of the range of A.
#     tol
#         Error tolerance for the algorithm.
#     failure_tolerance
#         Maximum failure probability.
#     num_testvecs
#         Number of test vectors.
#     lambda_min
#         The smallest eigenvalue of source_product.
#         If `None`, the smallest eigenvalue is computed using scipy.
#     train_vectors
#         |VectorArray| containing a set of predefined training
#         vectors.

#     Returns
#     -------
#     B
#         |VectorArray| which contains fine scale part of the computed solutions
#         of the oversampling problem.
#     """
#     problem = oversampling_problem

#     assert source_product is None or isinstance(source_product, Operator)
#     assert range_product is None or isinstance(range_product, Operator)
#     assert (
#         train_vectors is None
#         or isinstance(train_vectors, VectorArray)
#         and train_vectors.space is problem.source
#     )
#     assert coarse_scale_basis.space is problem.range

#     if source_product is None:
#         lambda_min = 1
#     elif lambda_min is None:

#         def mv(v):
#             return source_product.apply(source_product.source.from_numpy(v)).to_numpy()

#         def mvinv(v):
#             return source_product.apply_inverse(
#                 source_product.range.from_numpy(v)
#             ).to_numpy()

#         L = LinearOperator(
#             (source_product.source.dim, source_product.range.dim), matvec=mv
#         )
#         Linv = LinearOperator(
#             (source_product.range.dim, source_product.source.dim), matvec=mvinv
#         )
#         lambda_min = eigsh(
#             L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv
#         )[0]

#     # NOTE problem.source is the full space, while the source product is of lower dimension
#     num_source_dofs = len(problem._bc_dofs_gamma_out)
#     testfail = failure_tolerance / min(num_source_dofs, problem.range.dim)
#     testlimit = (
#         np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * tol
#     )
#     maxnorm = np.inf
#     # set of test vectors
#     R = problem.generate_random_boundary_data(num_testvecs, distribution="normal")
#     M = problem.solve(R)

#     # get vertex dofs of target subdomain
#     subdomain = problem.subdomain_problem.domain
#     vertices = subdomain.get_nodes(n=4)  # assumes rectangular domain
#     vertex_dofs = locate_dofs(problem.range.V.tabulate_dof_coordinates(), vertices)

#     logger.info(f"{lambda_min=}")
#     logger.info(f"{testlimit=}")

#     B = problem.range.empty()
#     snapshots = problem.range.empty()
#     while maxnorm > testlimit:
#         basis_length = len(B)
#         if train_vectors is not None and basis_length < len(train_vectors):
#             v = train_vectors[basis_length]
#         else:
#             v = problem.generate_random_boundary_data(count=1, distribution="normal")

#         r = problem.solve(v)
#         B.append(r.copy())

#         # subtract coarse scale part
#         nodal_values = r.dofs(vertex_dofs)
#         r -= coarse_scale_basis.lincomb(nodal_values)
#         snapshots.append(r)

#         gram_schmidt(
#             B,
#             range_product,
#             atol=0,
#             rtol=0,
#             offset=basis_length,
#             copy=False,
#         )
#         # requires B to be orthonormal wrt range_product
#         M -= B.lincomb(B.inner(M, range_product).T)

#         norm = M.norm(range_product)
#         if any(np.isnan(norm)):
#             breakpoint()
#         maxnorm = np.max(norm)

#     zero_at_dofs = np.allclose(
#         snapshots.to_numpy()[:, vertex_dofs], np.zeros(vertex_dofs.size)
#     )
#     assert zero_at_dofs

#     return snapshots


# # adapted from pymor.algorithms.randrangefinder.adaptive_rrf
# # to be used with multi.oversampling.OversamplingProblem
# def construct_spectral_basis(
#     logger,
#     oversampling_problem,
#     source_product=None,
#     range_product=None,
#     tol=1e-4,
#     failure_tolerance=1e-15,
#     num_testvecs=20,
#     lambda_min=None,
#     train_vectors=None,
# ):
#     r"""Adaptive randomized range approximation of `A`.
#     This is an implementation of Algorithm 1 in [BS18]_.

#     The following modifications where made:
#         - instead of a transfer operator A, the full oversampling
#         problem is used

#     The image Av = A.apply(v) is equivalent to the restriction
#     of the full solution to the target domain Ω_in, i.e.
#         U = oversampling_problem.solve(v)
#     See multi.oversampling.OversamplingProblem.solve.

#     Given the |Operator| `A`, the return value of this method is the |VectorArray|
#     `B` with the property

#     .. math::
#         \Vert A - P_{span(B)} A \Vert \leq tol

#     with a failure probability smaller than `failure_tolerance`, where the norm denotes the
#     operator norm. The inner product of the range of `A` is given by `range_product` and
#     the inner product of the source of `A` is given by `source_product`.

#     Parameters
#     ----------
#     logger
#         The logger used by the main program.
#     oversampling_problem
#         The full oversampling problem associated with a (transfer) |Operator| A.
#     source_product
#         Inner product |Operator| of the source of A.
#     range_product
#         Inner product |Operator| of the range of A.
#     tol
#         Error tolerance for the algorithm.
#     failure_tolerance
#         Maximum failure probability.
#     num_testvecs
#         Number of test vectors.
#     lambda_min
#         The smallest eigenvalue of source_product.
#         If `None`, the smallest eigenvalue is computed using scipy.
#     train_vectors
#         |VectorArray| containing a set of predefined training
#         vectors.

#     Returns
#     -------
#     B
#         |VectorArray| which contains the spectral basis.
#     """
#     problem = oversampling_problem

#     assert source_product is None or isinstance(source_product, Operator)
#     assert range_product is None or isinstance(range_product, Operator)
#     assert (
#         train_vectors is None
#         or isinstance(train_vectors, VectorArray)
#         and train_vectors.space is problem.source
#     )

#     if source_product is None:
#         lambda_min = 1
#     elif lambda_min is None:

#         def mv(v):
#             return source_product.apply(source_product.source.from_numpy(v)).to_numpy()

#         def mvinv(v):
#             return source_product.apply_inverse(
#                 source_product.range.from_numpy(v)
#             ).to_numpy()

#         L = LinearOperator(
#             (source_product.source.dim, source_product.range.dim), matvec=mv
#         )
#         Linv = LinearOperator(
#             (source_product.range.dim, source_product.source.dim), matvec=mvinv
#         )
#         lambda_min = eigsh(
#             L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv
#         )[0]

#     # NOTE problem.source is the full space, while the source product is of lower dimension
#     num_source_dofs = len(problem._bc_dofs_gamma_out)
#     testfail = failure_tolerance / min(num_source_dofs, problem.range.dim)
#     testlimit = (
#         np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * tol
#     )
#     maxnorm = np.inf
#     # set of test vectors
#     R = problem.generate_random_boundary_data(num_testvecs, distribution="normal")
#     M = problem.solve(R)

#     logger.info(f"{lambda_min=}")
#     logger.info(f"{testlimit=}")

#     B = problem.range.empty()
#     while maxnorm > testlimit:
#         basis_length = len(B)
#         if train_vectors is not None and basis_length < len(train_vectors):
#             v = train_vectors[basis_length]
#         else:
#             v = problem.generate_random_boundary_data(count=1, distribution="normal")
#         B.append(problem.solve(v))

#         gram_schmidt(
#             B,
#             range_product,
#             atol=0,
#             rtol=0,
#             offset=basis_length,
#             copy=False,
#         )
#         # requires B to be orthonormal wrt range_product
#         M -= B.lincomb(B.inner(M, range_product).T)

#         norm = M.norm(range_product)
#         if any(np.isnan(norm)):
#             breakpoint()
#         maxnorm = np.max(norm)

#     return B
