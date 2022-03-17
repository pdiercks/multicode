import dolfin as df


# adapted from pymor.bindings.fenics.py
# https://github.com/pymor/pymor/blob/efcde8e2007e90b61fc68266c432cc9175aceacb/src/pymor/bindings/fenics.py
_DEFAULT_SOLVER = "mumps" if "mumps" in df.linear_solver_methods() else "default"


def _solver_options(solver=_DEFAULT_SOLVER, preconditioner=None, keep_solver=True):
    return {
        "solver": solver,
        "preconditioner": preconditioner,
        "keep_solver": keep_solver,
    }


def create_solver(matrix, solver_options=_solver_options()):
    options = solver_options
    method = options.get("solver")
    preconditioner = options.get("preconditioner")
    if method == "lu" or method in df.lu_solver_methods():
        method = "default" if method == "lu" else method
        solver = df.LUSolver(matrix, method)
    else:
        solver = df.KrylovSolver(matrix, method, preconditioner)
    return solver
