"""build nullspace"""

import pytest

from mpi4py import MPI
from dolfinx import fem, mesh
from basix.ufl import element

from pymor.bindings.fenicsx import FenicsxVectorSpace
from multi.solver import build_nullspace


@pytest.mark.parametrize("gdim", [1, 2, 3])
def test(gdim):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    fe = element("P", domain.basix_cell(), 1, shape=(gdim,))
    V = fem.functionspace(domain, fe)
    source = FenicsxVectorSpace(V)

    ns = []
    if gdim == 1:
        with pytest.raises(NotImplementedError):
            build_nullspace(source, product=None, gdim=gdim)
    else:
        ns = build_nullspace(source, product=None, gdim=gdim) 
    if gdim == 2:
        assert len(ns) == 3

    if gdim == 3:
        assert len(ns) == 6
