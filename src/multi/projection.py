"""projection module"""

from typing import Union, Optional
import numpy as np
from dolfinx import fem
from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray
from pymor.algorithms.basic import project_array, relative_error


def compute_relative_proj_errors(U: VectorArray, basis: VectorArray, product: Optional[Operator] = None, orthonormal: bool = True) -> list[float]:
    """Compute relative projection errors.

    Args:
        U: The test data.
        basis: Basis to project onto.
        product: Inner product to use.
        orthonormal: If True, assume basis is orthonormal wrt product.
    """
    errors = []
    for N in range(len(basis) + 1):
        U_proj = project_array(U, basis[:N], product=product, orthonormal=orthonormal)
        relerr = relative_error(U, U_proj, product=product)
        errors.append(np.max(relerr))
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
