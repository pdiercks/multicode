import dolfinx
import numpy as np
from mpi4py import MPI
from multi.domain import RceDomain
from multi.problems import LinearElasticityProblem
from multi.basis_construction import compute_phi
from multi.misc import locate_dofs
from pymor.bindings.fenicsx import FenicsxVectorSpace


def xdofs_VectorFunctionSpace(V):
    bs = V.dofmap.bs
    x = V.tabulate_dof_coordinates()
    x_dofs = np.repeat(x, repeats=bs, axis=0)
    return x_dofs


def test():
    n = 2
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 2))
    rce = RceDomain(domain, edges=True)
    problem = LinearElasticityProblem(rce, V, E=60e3, NU=0.2, plane_stress=True)
    phi = compute_phi(problem)
    source = FenicsxVectorSpace(V)
    B = source.make_array(phi)

    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]) 
    breakpoint()
    vertex_dofs = locate_dofs(xdofs_VectorFunctionSpace(V), vertices, gdim=3)
    nodal_values = B.dofs(vertex_dofs)
    assert len(B) == 8

    assert np.sum(nodal_values) == 8


if __name__ == "__main__":
    test()
