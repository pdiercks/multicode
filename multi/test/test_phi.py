import numpy as np
import dolfin as df
from multi import RectangularDomain, LinearElasticityProblem
from multi.basis_construction import compute_phi
from multi.misc import locate_dofs


def test():
    mesh = df.UnitSquareMesh(10, 10)
    V = df.VectorFunctionSpace(mesh, "CG", 2)
    domain = RectangularDomain(mesh)
    problem = LinearElasticityProblem(domain, V, E=60e3, NU=0.2, plane_stress=True)
    solver_options = {"inverse": {"solver": "mumps"}}
    phi = compute_phi(problem, solver_options=solver_options)

    vertices = domain.get_nodes(n=4)
    vertex_dofs = locate_dofs(V.tabulate_dof_coordinates(), vertices)
    nodal_values = phi.dofs(vertex_dofs)
    assert len(phi) == 8
    assert np.sum(nodal_values) == 8


if __name__ == "__main__":
    test()
