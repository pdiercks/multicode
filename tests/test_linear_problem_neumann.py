from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from basix.ufl import element
import ufl
import numpy as np

from multi.domain import Domain
from multi.problems import LinearProblem


class TestProblem(LinearProblem):
    def __init__(self, domain, V):
        super().__init__(domain, V)

    @property
    def form_rhs(self):
        domain = self.domain.grid
        v = self.test
        f = fem.Constant(domain, default_scalar_type(-6))
        rhs = f * v * ufl.dx

        if self._bc_handler.has_neumann:
            rhs += self._bc_handler.neumann_bcs

        return rhs

    @property
    def form_lhs(self):
        u = self.trial
        v = self.test
        return ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx


def test():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    Ω = Domain(domain)
    fe = element("P", domain.basix_cell(), 1, shape=())
    V = fem.functionspace(domain, fe)

    problem = TestProblem(Ω, V)

    def u_exact(x):
        return 1 + x[0] ** 2 + 2 * x[1] ** 2

    def boundary_D(x):
        return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))

    # dirichlet
    u_bc = fem.Function(V)
    u_bc.interpolate(u_exact)
    problem.add_dirichlet_bc(u_bc, boundary_D, method="geometrical")

    # neumann
    x = ufl.SpatialCoordinate(domain)
    g = 4 * x[1]  # NOTE sign of term on the rhs
    # since no markers are defined simply pass 'everywhere'
    problem.add_neumann_bc("everywhere", g)

    # setup the solver
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem.setup_solver(petsc_options=petsc_options)
    uh = problem.solve()

    uex = fem.Function(V)
    uex.interpolate(u_exact)

    error_max = np.max(np.abs(uex.x.array[:] - uh.x.array[:]))
    error_max = MPI.COMM_WORLD.allreduce(error_max, op=MPI.MAX)
    assert error_max < 1e-12


if __name__ == "__main__":
    test()
