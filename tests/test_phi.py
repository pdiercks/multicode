from mpi4py import MPI
from dolfinx import mesh, fem
from basix.ufl import element
import numpy as np
from multi.domain import RectangularSubdomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.basis_construction import compute_phi
from multi.misc import locate_dofs
from pymor.bindings.fenicsx import FenicsxVectorSpace


def xdofs_VectorFunctionSpace(V):
    bs = V.dofmap.bs
    x = V.tabulate_dof_coordinates()
    x_dofs = np.repeat(x, repeats=bs, axis=0)
    return x_dofs


def test_nodes():
    n = 20
    domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    ve = element("P", domain.basix_cell(), 2, shape=(2,))
    V = fem.functionspace(domain, ve)

    rce = RectangularSubdomain(1, domain)
    rce.create_edge_grids({"fine": 20})
    gdim = domain.ufl_cell().geometric_dimension()

    phases = (LinearElasticMaterial(gdim, 60e3, 0.2, plane_stress=True),)
    problem = LinearElasticityProblem(rce, V, phases)

    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    vertex_dofs = locate_dofs(xdofs_VectorFunctionSpace(V), vertices, gdim=3)

    phi = compute_phi(problem, vertices)
    source = FenicsxVectorSpace(V)
    B = source.make_array(phi)
    nodal_values = B.dofs(vertex_dofs)

    assert len(B) == 8
    barr = B.to_numpy()
    assert np.isclose(np.sum(nodal_values), 8)
    assert not np.allclose(barr[0], barr[1])


if __name__ == "__main__":
    test_nodes()
