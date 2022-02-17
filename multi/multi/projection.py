# Authorship: the pyMOR developers
# the original code is part of the pymor tutorial https://docs.pymor.org/2020.2.0/tutorial_basis_generation.html
import numpy as np


def compute_proj_errors(basis, V, product):
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


def compute_proj_errors_orth_basis(basis, V, product):
    errors = []
    for N in range(len(basis) + 1):
        v = V.inner(basis[:N], product=product)
        V_proj = basis[:N].lincomb(v)
        err = (V - V_proj).norm(product=product)
        alpha = V.norm(product=product)
        err /= alpha
        errors.append(np.max(err))
    return errors
