import dolfinx
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from multi.domain import Domain
from multi.problems import LinearProblem


class TestProblem(LinearProblem):
    def __init__(self, domain, V, solver_options=None):
        super().__init__(domain, V, solver_options)

    def get_form_rhs(self):
        domain = self.domain.mesh
        v = self.v
        f = dolfinx.fem.Constant(domain, PETSc.ScalarType(-6))
        rhs = f * v * ufl.dx

        if self._bc_handler.has_neumann:
            rhs += self._bc_handler.neumann_bcs

        return rhs

    def get_form_lhs(self):
        u = self.u
        v = self.v
        return ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx


def test():
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    Ω = Domain(domain)
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))

    problem = TestProblem(Ω, V)

    def u_exact(x):
        return 1 + x[0] ** 2 + 2 * x[1] ** 2

    def boundary_D(x):
        return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))

    # dirichlet
    u_bc = dolfinx.fem.Function(V)
    u_bc.interpolate(u_exact)
    problem.add_dirichlet_bc(u_bc, boundary_D, method="geometrical")

    # neumann
    x = ufl.SpatialCoordinate(domain)
    g = 4 * x[1]  # NOTE sign of term on the rhs
    # since no markers are defined simply pass 'everywhere'
    problem.add_neumann_bc("everywhere", g)

    uh = problem.solve()

    uex = dolfinx.fem.Function(V)
    uex.interpolate(u_exact)

    error_max = np.max(np.abs(uex.x.array[:] - uh.x.array[:]))
    error_max = MPI.COMM_WORLD.allreduce(error_max, op=MPI.MAX)
    assert error_max < 1e-12


if __name__ == "__main__":
    test()
