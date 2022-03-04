import dolfin as df

# Test for PETSc
if not df.has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

# Set backend to PETSC
prm = df.parameters
prm["linear_algebra_backend"] = "PETSc"


def extend(
    problem,
    boundary_data,
    solver_options={"linear_solver": "default", "preconditioner": "default"},
):
    """extend boundary data into domain associated with the given problem

    Parameters
    ----------
    problem
        The variational problem.
    boundary_data
        A list of dolfin functions (elements of problem.V).

    Returns
    -------
    A list of dolfin vectors.
    """

    # assemble lhs
    A = df.PETScMatrix()
    lhs_form = problem.get_lhs()
    df.assemble(lhs_form, tensor=A)

    # rhs
    B = A.copy()

    # prepare operator
    V = problem.V
    dummy = df.Function(V)
    bc = df.DirichletBC(V, dummy, df.DomainBoundary())
    bc.zero_columns(A, dummy.vector(), 1.0)

    # solver
    method = solver_options["linear_solver"]
    preconditioner = solver_options["preconditioner"]
    solver = df.KrylovSolver(A, method, preconditioner)

    # Vectors for solution x, boundary data g, rhs b
    b = df.Function(V)

    extended = []
    for g in boundary_data:
        x = df.Function(V)
        bc = df.DirichletBC(V, g, df.DomainBoundary())
        b.vector().zero()
        bc.zero_columns(B.copy(), b.vector(), 1.0)
        solver.solve(x.vector(), b.vector())
        extended.append(x.vector())

    return extended
