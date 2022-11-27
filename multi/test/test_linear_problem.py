import numpy as np
import ufl
import dolfinx
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
from multi.domain import Domain
from multi.problems import LinearProblem


class PoissonProblem(LinearProblem):
    def __init__(self, domain, V):
        super().__init__(domain, V)
        self.dx = ufl.dx

    @property
    def form_lhs(self):
        u = self.u
        v = self.v
        return ufl.dot(ufl.grad(u), ufl.grad(v)) * self.dx

    @property
    def form_rhs(self):
        f = dolfinx.fem.Constant(self.domain.grid, ScalarType(-6))
        return f * self.v * self.dx


def test_poisson():
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, dolfinx.mesh.CellType.quadrilateral
    )
    Ω = Domain(domain)
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
    # instantiate problem class
    problem = PoissonProblem(Ω, V)

    # add a Dirichlet bc
    uD = dolfinx.fem.Function(V)
    uD.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
    # Create facet to cell connectivity required to determine boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    problem.add_dirichlet_bc(uD, boundary_facets, entity_dim=fdim)

    # setup the solver
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem.setup_solver(petsc_options=petsc_options)

    uh = problem.solve()

    error_max = np.max(np.abs(uD.x.array - uh.x.array))
    assert error_max < 1e-12


if __name__ == "__main__":
    test_poisson()
