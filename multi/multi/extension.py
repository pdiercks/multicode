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
    pc = df.PETScPreconditioner(solver_options["preconditioner"])
    solver = df.PETScKrylovSolver(solver_options["linear_solver"], pc)
    solver.set_operator(A)
    solver.set_reuse_preconditioner(True)

    # Vectors for solution x, boundary data g, rhs b
    # x = df.Function(V)
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


# # FIXME
# # scipy implementation is faster for smaller problems (~10k DoFs)
# # BUT I have no means to control the solver accuracy
# def scipy_extend_function(
#     problem, boundary_data, solver_options=None
# ):
#     """extend (boundary/edge) function into interior of the domain

#     Parameters
#     ----------
#     problem
#         The linear variational problem.
#     boundary_function : np.ndarray
#         The dofs of the boundary function.
#     V_to_L : int
#         The map from V (domain space) to L (boundary/edge space).
#     solver_options : optional, dict
#         Solver options.

#     Returns
#     -------
#     psi
#         The VectorArray containing basis functions.

#     """

#     V = problem.V
#     one = df.Function(V)
#     one.vector().set_local(np.ones(V.dim()))
#     bcs = df.DirichletBC(V, one, df.DomainBoundary())
#     dirichlet_dofs = np.array(list(bcs.get_boundary_values().keys()))
#     A = df.assemble(problem.get_lhs())
#     Amat = df.as_backend_type(A).mat()
#     full_operator = csc_matrix(Amat.getValuesCSR()[::-1], shape=Amat.size)

#     B = A.copy()
#     dummy = df.Function(V)
#     bcs.zero_columns(B, dummy.vector(), 1.0)
#     Bmat = df.as_backend_type(B).mat()
#     operator = csc_matrix(Bmat.getValuesCSR()[::-1], shape=Bmat.size)

#     # factorization
#     matrix_shape = operator.shape
#     start = time.time()
#     operator = factorized(operator)
#     end = time.time()
#     print(f"factorization of {matrix_shape} matrix in {end-start}")

#     rhs_op = full_operator[:, dirichlet_dofs]
#     rhs_op[dirichlet_dofs, :] = np.zeros(dirichlet_dofs.size)
#     start = time.time()
#     final_operator = -operator(rhs_op.todense())
#     end = time.time()
#     print(f"applied operator to rhs in {end-start}")

#     extended = []
#     for g in boundary_data:
#         d = final_operator.dot(g.vector().get_local()[dirichlet_dofs])
#         d[0, dirichlet_dofs] += g.vector().get_local()[dirichlet_dofs]
#         extended.append(np.array(d))

#     return np.vstack(extended)
