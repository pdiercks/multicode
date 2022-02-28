import numpy as np
import dolfin as df
from multi.extension import extend

# Test for PETSc
if not df.has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

# Set backend to PETSC
prm = df.parameters
prm["linear_algebra_backend"] = "PETSc"


def test():
    mesh = df.UnitSquareMesh(6, 6)
    V = df.FunctionSpace(mesh, "CG", 2)

    class DummyProblem:
        def __init__(self, V, lhs):
            self.V = V
            self.lhs = lhs

        def get_lhs(self):
            return self.lhs

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    lhs = df.inner(df.grad(u), df.grad(v)) * df.dx
    rhs = df.Constant(0.0) * v * df.dx
    g = df.interpolate(df.Expression("x[0] - x[0] * x[0]", degree=2), V)
    f = df.interpolate(df.Expression("x[1] - x[1] * x[1]", degree=2), V)
    bc = df.DirichletBC(V, g, df.DomainBoundary())
    u_ref = df.Function(V)

    A, b = df.assemble_system(lhs, rhs, bc)
    pc = df.PETScPreconditioner("default")
    solver = df.PETScKrylovSolver("default", pc)
    solver.set_operator(A)
    solver.solve(u_ref.vector(), b)

    problem = DummyProblem(V, lhs)
    s = extend(problem, [g, f])

    err = df.Function(V)
    err.vector().axpy(1.0, u_ref.vector())
    err.vector().axpy(-1.0, s[0])
    norm = np.linalg.norm(err.vector()[:])
    assert norm < 1e-9

    err.vector().zero()
    err.vector().axpy(1.0, u_ref.vector())
    err.vector().axpy(-1.0, s[1])
    norm = np.linalg.norm(err.vector()[:])
    assert not norm < 1e-9


if __name__ == "__main__":
    test()
