"""projection module"""

# NOTE regarding functions compute_proj_errors, project and compute_proj_errors_orth_basis
# Authorship: the pyMOR developers
# the original code is part of the pymor tutorial https://docs.pymor.org/2020.2.0/tutorial_basis_generation.html
import numpy as np
from dolfinx.fem import Function


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


def orthogonal_part(basis, V, product, orth=False):
    """return part of V that is orthogonal to span(basis)"""
    V_proj = project(basis, V, product, orth=orth)
    return V - V_proj


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


def fine_scale_part(u, coarse_space, in_place=False):
    """returns fine scale part u_f = u - u_c

    Parameters
    ----------
    u : dolfinx.fem.Function
        The function whose fine scale part is computed.
    coarse_space : dolfinx.fem.FunctionSpace
        The coarse FE space W (u_c in W).
    in_place : optional, bool
        If True, modify u in-place.

    Returns
    -------
    u_f: dolfinx.fem.Function or None if `in_place==True`.

    Note
    ----
    u.function_space.mesh and coarse_space.mesh need to be
    partitions of the same domain Ω.
    """

    V = u.function_space
    u_c = Function(V)
    w = Function(coarse_space)

    w.interpolate(u)
    u_c.interpolate(w)

    if in_place:
        u.vector.axpy(-1, u_c.vector)
    else:
        u_f = Function(V)
        u_f.vector.axpy(1.0, u.vector)
        u_f.vector.axpy(-1.0, u_c.vector)
        return u_f
