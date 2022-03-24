"""projection module"""

# NOTE regarding functions compute_proj_errors, project and compute_proj_errors_orth_basis
# Authorship: the pyMOR developers
# the original code is part of the pymor tutorial https://docs.pymor.org/2020.2.0/tutorial_basis_generation.html
import numpy as np
from dolfin import Function, DirichletBC
from multi.shapes import NumpyQuad
from pymor.algorithms.pod import pod
from pymor.bindings.fenics import FenicsVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


def compute_proj_errors(basis, V, product, relative=True):
    G = basis.gramian(product=product)
    R = basis.inner(V, product=product)
    errors = []
    for N in range(len(basis) + 1):
        if N > 0:
            try:
                v = np.linalg.solve(G[:N, :N], R[:N, :])
            except np.linalg.LinAlgError:
                break
            except np.any(np.isnan(v)):
                break
        else:
            # N = 0, such that err is the norm of V
            v = np.zeros((0, len(V)))
        V_proj = basis[:N].lincomb(v.T)
        err = (V - V_proj).norm(product=product)
        if np.any(np.isnan(err)):
            print("---- OOOOPS! ---- ")
            print("---- NaN for product {}, N = {} ----".format(product.name, N))
            break
        if relative:
            alpha = V.norm(product=product)
            err /= alpha
        errors.append(np.max(err))
    return errors


def project(basis, V, product, orth=False):
    """project V onto basis"""
    if orth:
        v = V.inner(basis, product=product)
        V_proj = basis.lincomb(v)
    else:
        G = basis.gramian(product=product)
        R = basis.inner(V, product=product)
        v = np.linalg.solve(G, R)
        V_proj = basis.lincomb(v.T)

    return V_proj


def compute_proj_errors_orth_basis(basis, V, product, relative=True):
    errors = []
    for N in range(len(basis) + 1):
        v = V.inner(basis[:N], product=product)
        V_proj = basis[:N].lincomb(v)
        err = (V - V_proj).norm(product=product)
        if relative:
            alpha = V.norm(product=product)
            err /= alpha
        errors.append(np.max(err))
    return errors


def project_soi_data(
    data_path, domain, V, restrict_to_boundary=None, soi_degree=2, use_pod=False
):
    """project structure of interest data into space

    Parameters
    ----------
    data_path : FilePath
        The path to the structure of interest data.
    domain : multi.RectangularDomain
        The computational domain. Note that this has to
        be a rectangular domain.
    V : dolfin.FunctionSpace
        The function space to project to.
    restrict_to_boundary : dolfin.SubDomain, optional
        Restrict the projection to a given boundary of the domain.
    soi_degree : int, optional
        Polynomial degree with which the SoI data was computed.
    use_pod : bool, optional
        If True, use pod to filter the SoI data.

    Returns
    -------
    numpy.ndarray
        The projected SoI data.
    """
    # prepare global shape functions in space
    if soi_degree == 1:
        num_nodes = 4
    elif soi_degree == 2:
        num_nodes = 9
    else:
        raise NotImplementedError
    nodes = domain.get_nodes(n=num_nodes)
    quadrilateral = NumpyQuad(nodes)
    shape_functions = quadrilateral.interpolate(V)
    source = FenicsVectorSpace(V)
    shapes = source.from_numpy(shape_functions)

    # prepare SoI data
    data_array = np.load(data_path)
    numpy_space = NumpyVectorSpace(data_array.shape[1])
    soi_data = numpy_space.make_array(data_array)
    if use_pod:
        soi_data, svals = pod(soi_data)
    U = shapes.lincomb(soi_data.to_numpy())

    if restrict_to_boundary is not None:
        # restrict U to boundary
        boundary = restrict_to_boundary
        bc = DirichletBC(V, Function(V), boundary)
        boundary_dofs = list(bc.get_boundary_values().keys())
        return U.dofs(boundary_dofs)
    else:
        return U.to_numpy()
