from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, mesh, default_scalar_type
from basix.ufl import element
from multi.domain import Domain
from multi.problems import LinearProblem


class PoissonProblem(LinearProblem):
    def __init__(self, domain, V):
        super().__init__(domain, V)
        self.dx = ufl.dx

    @property
    def form_lhs(self):
        u = self.trial
        v = self.test
        return ufl.dot(ufl.grad(u), ufl.grad(v)) * self.dx

    @property
    def form_rhs(self):
        f = fem.Constant(self.domain.grid, default_scalar_type(-6))
        return f * self.test * self.dx


def test_poisson():
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral
    )
    Ω = Domain(domain)
    fe = element("P", domain.basix_cell(), 1, shape=())
    V = fem.functionspace(domain, fe)
    # instantiate problem class
    problem = PoissonProblem(Ω, V)

    # add a Dirichlet bc
    uD = fem.Function(V)
    uD.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
    # Create facet to cell connectivity required to determine boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    problem.add_dirichlet_bc(uD, boundary_facets, entity_dim=fdim)

    # setup the solver
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem.setup_solver(petsc_options=petsc_options)

    bcs = problem.bcs
    problem.assemble_matrix(bcs)
    problem.assemble_vector(bcs)
    solver = problem.solver
    uh = problem.u
    solver.solve(problem.b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    error_max = np.max(np.abs(uD.x.array - uh.x.array))
    assert error_max < 1e-12


if __name__ == "__main__":
    test_poisson()
