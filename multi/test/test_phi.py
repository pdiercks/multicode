import numpy as np
import dolfin as df
from multi import Domain, LinearElasticityProblem
from multi.basis_construction import compute_phi
from multi.misc import locate_dofs

# Test for PETSc
if not df.has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

# Set backend to PETSC
prm = df.parameters
prm["linear_algebra_backend"] = "PETSc"
solver_prm = prm["krylov_solver"]
solver_prm["relative_tolerance"] = 1e-9
solver_prm["absolute_tolerance"] = 1e-12


def test():
    mesh = df.UnitSquareMesh(10, 10)
    V = df.VectorFunctionSpace(mesh, "CG", 2)
    domain = Domain(mesh)
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
