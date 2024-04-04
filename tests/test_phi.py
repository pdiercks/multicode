from mpi4py import MPI
from dolfinx import mesh, fem
from basix.ufl import element
import numpy as np
from multi.domain import RectangularSubdomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.basis_construction import compute_phi
from multi.misc import locate_dofs, x_dofs_vectorspace
from pymor.bindings.fenicsx import FenicsxVectorSpace


def test():
    n = 20
    domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    ve = element("P", domain.basix_cell(), 2, shape=(2,))
    V = fem.functionspace(domain, ve)

    rce = RectangularSubdomain(1, domain)
    rce.create_coarse_grid(1)
    rce.create_boundary_grids()
    gdim = domain.geometry.dim

    phases = LinearElasticMaterial(gdim, 60e3, 0.2, plane_stress=True)
    problem = LinearElasticityProblem(rce, V, phases)

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    vertex_dofs = locate_dofs(x_dofs_vectorspace(V), vertices)

    phi = compute_phi(problem, vertices)
    source = FenicsxVectorSpace(V)
    B = source.make_array(phi)
    nodal_values = B.dofs(vertex_dofs)

    assert len(B) == 8
    barr = B.to_numpy()
    assert np.isclose(np.sum(nodal_values), 8)
    assert not np.allclose(barr[0], barr[1])


if __name__ == "__main__":
    test()
