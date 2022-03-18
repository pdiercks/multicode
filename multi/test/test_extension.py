import numpy as np
import dolfin as df
from multi.solver import create_solver
from multi.extension import extend, extend_pymor
from pymor.bindings.fenics import FenicsMatrixOperator, FenicsVectorSpace


def test_rhs():
    mesh = df.UnitSquareMesh(50, 50)
    V = df.FunctionSpace(mesh, "CG", 2)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    lhs = df.inner(df.grad(u), df.grad(v)) * df.dx
    A = df.assemble(lhs)

    def boundary_expression():
        return "x[0] - x[0] * x[0]"

    bc = df.DirichletBC(
        V, df.Expression(boundary_expression(), degree=2), df.DomainBoundary()
    )

    # ### compute using pymor
    B = FenicsMatrixOperator(
        A.copy(), V, V, solver_options={"inverse": {"solver": "mumps"}}, name="B"
    )
    g = df.Function(V)  # boundary data
    gvec = g.vector()
    gvec.zero()
    bc.apply(gvec)
    G = B.source.make_array([gvec])
    rhs = -B.apply(G)
    bc_dofs = list(bc.get_boundary_values().keys())
    bc_vals = list(bc.get_boundary_values().values())
    rhs_array = rhs.to_numpy()
    rhs_array[:, bc_dofs] = np.array(bc_vals)

    # ### compute with bc.zero_columns
    reference = df.Function(V)
    bc.zero_columns(A, reference.vector(), 1.0)

    # ### error
    err = reference.vector().get_local() - rhs_array.flatten()
    assert np.linalg.norm(err) < 1e-9


def test():
    num_cells = 20
    mesh = df.UnitSquareMesh(num_cells, num_cells)
    V = df.FunctionSpace(mesh, "CG", 2)
    print(f"Number of DoFs={V.dim()}")

    class DummyProblem:
        def __init__(self, V, lhs):
            self.V = V
            self.lhs = lhs
            self.source = FenicsVectorSpace(V)

        def get_lhs(self):
            return self.lhs

    def boundary_expression():
        return "x[0] - x[0] * x[0]"

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    lhs = df.inner(df.grad(u), df.grad(v)) * df.dx
    rhs = df.Constant(0.0) * v * df.dx
    bcs = []
    num_test = 100
    for i in range(num_test):
        bc = df.DirichletBC(
            V, df.Expression(boundary_expression(), degree=2), df.DomainBoundary()
        )
        bcs.append(bc)
    u_ref = df.Function(V)

    A, b = df.assemble_system(lhs, rhs, bcs[0])
    if V.dim() < 10e3:
        options = {"solver": "mumps"}
    else:
        options = {"solver": "cg", "preconditioner": "petsc_amg"}
        solver_prm = df.parameters["krylov_solver"]
        solver_prm["absolute_tolerance"] = 1e-12
        solver_prm["relative_tolerance"] = 1e-9
    print("solver options", options)
    solver = create_solver(A, solver_options=options)
    solver.solve(u_ref.vector(), b)

    # ### prepare boundary data in V, such that
    # boundary_data[bc_dofs] == bc_vals
    # boundary_data[inner_dofs] == 0.0
    # inner_dofs = setdiff1d(all_dofs, bc_dofs)
    all_dofs = np.arange(V.dim())
    boundary_vectors = []
    boundary_functions = []
    for bc in bcs:
        bc_dofs = list(bc.get_boundary_values().keys())
        bc_vals = list(bc.get_boundary_values().values())
        inner_dofs = np.setdiff1d(all_dofs, bc_dofs)
        g = df.Function(V)
        gvec = g.vector()
        gvec.zero()
        bc.apply(gvec)
        assert np.allclose(g.vector()[bc_dofs], bc_vals)
        assert np.allclose(
            g.vector()[inner_dofs], np.zeros_like(g.vector()[inner_dofs])
        )
        boundary_vectors.append(gvec)
        boundary_functions.append(g)

    problem = DummyProblem(V, lhs)
    s = extend(problem, boundary_functions, solver_options=options)

    U = extend_pymor(problem, boundary_vectors, solver_options={"inverse": options})
    err = u_ref.vector()[:] - U.to_numpy()[0]
    norm = np.linalg.norm(err)
    print(f"norm pymor version {norm}")
    assert norm < 1e-9

    err = df.Function(V)
    err.vector().axpy(1.0, u_ref.vector())
    err.vector().axpy(-1.0, s[0])
    norm = np.linalg.norm(err.vector()[:])
    print(f"norm pure fenics version {norm}")
    assert norm < 1e-9

    # ### Profiling Results
    # usage: kernprof -lv __file__ with num_test=100 and varying num_cells
    # num_test | num_cells | Dofs  | method | preconditioner | extend   | extend_pymor
    # 100      | 20        | 1681  | mumps  | -              | 0.19612s | 0.112597s
    # 100      | 50        | 10201 | cg     | petsc_amg      | 3.46028s | 3.04381s
    # 100      | 80        | 25921 | cg     | petsc_amg      | 8.66885s | 7.88604s


if __name__ == "__main__":
    test_rhs()
    test()
