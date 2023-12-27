"""projection module"""

# NOTE regarding functions compute_proj_errors and compute_proj_errors_orth_basis
# Authorship: the pyMOR developers
# the original code is part of the pymor tutorial https://docs.pymor.org/2020.2.0/tutorial_basis_generation.html
from typing import Union, Optional
import numpy as np
from dolfinx import fem
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray
from pymor.algorithms.basic import project_array


def compute_proj_errors(basis: VectorArray, V: VectorArray, product: Operator, relative: bool = True):
    """Computes projection error for non-orthogonal basis"""
    G = basis.gramian(product=product)
    R = basis.inner(V, product=product)
    errors = []
    for N in range(len(basis) + 1):
        if N > 0:
            v = np.linalg.solve(G[:N, :N], R[:N, :])
        else:
            # N = 0, such that err is the norm of V
            v = np.zeros((0, len(V)))
        V_proj = basis[:N].lincomb(v.T)
        err = (V - V_proj).norm(product=product)
        if relative:
            alpha = V.norm(product=product)
            err /= alpha
        errors.append(np.max(err))
    return errors


def orthogonal_part(U: VectorArray, basis: VectorArray, product: Optional[Operator] = None, orthonormal: bool = True):
    """Returns part of V that is orthogonal to span(basis).

    Args:
        U: The VectorArray to project.
        basis: The basis to project onto.
        product: The inner product to use.
        orthonormal: If the basis is orthonormal wrt product.

    """
    U_proj = project_array(U, basis, product=product, orthonormal=orthonormal)
    return U - U_proj


def compute_proj_errors_orth_basis(basis: VectorArray, V: VectorArray, product: Operator, relative: bool = True):
    """Computes projection error for orthonormal basis"""
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


def fine_scale_part(u: fem.Function, coarse_space: fem.FunctionSpaceBase, in_place: bool = False) -> Union[fem.Function, None]:
    """Computes fine scale part u_f = u - u_c.

    Args:
        u: The function whose fine scale part should be computed.
        coarse_space: The coarse FE space W (u_c in W).
        in_place: If True, use `axpy` to subtract coarse scale part.

    Returns:
        u_f: The fine scale part.

    """

    V = u.function_space
    u_c = fem.Function(V)
    w = fem.Function(coarse_space)

    w.interpolate(u, nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
        w.function_space.mesh._cpp_object,
        w.function_space.element,
        u.function_space.mesh._cpp_object))
    u_c.interpolate(w, nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
        u_c.function_space.mesh._cpp_object,
        u_c.function_space.element,
        w.function_space.mesh._cpp_object))

    if in_place:
        u.vector.axpy(-1, u_c.vector)
    else:
        u_f = fem.Function(V)
        u_f.vector.axpy(1.0, u.vector)
        u_f.vector.axpy(-1.0, u_c.vector)
        return u_f
