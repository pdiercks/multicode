import numpy as np
import dolfin as df
from pymor.bindings.fenics import FenicsMatrixOperator


def extend_pymor(
    problem,
    boundary_data,
    solver_options={"inverse": {"solver": "mumps"}},
):
    """extend boundary data into domain associated with the given problem

    Parameters
    ----------
    problem
        The variational problem.
    boundary_data
        A list of dolfin vectors (elements of problem.V),
        a numpy array or a VectorArray.
    solver_options : dict, optional
        Options in pymor format.

    Returns
    -------
    U : VectorArray
        The extended vectors.
    """

    # assemble lhs
    A = df.PETScMatrix()
    lhs_form = problem.get_form_lhs()
    df.assemble(lhs_form, tensor=A)

    # rhs
    V = problem.V
    B = FenicsMatrixOperator(A.copy(), V, V, solver_options=solver_options, name="B")

    # prepare operator
    dummy = df.Function(V)
    bc = df.DirichletBC(V, dummy, df.DomainBoundary())
    bc.zero_columns(A, dummy.vector(), 1.0)
    # see FenicsMatrixOperator._real_apply_inverse_one_vector
    A = FenicsMatrixOperator(A, V, V, solver_options=solver_options, name="A")

    # wrap boundary_data as FenicsVectorArray
    space = B.range
    if isinstance(boundary_data, np.ndarray):
        R = space.from_numpy(boundary_data)
    # elif isinstance(boundary_data, list):
    #     R = space.make_array(boundary_data)
    else:
        R = space.make_array(boundary_data)

    # form rhs for each problem
    # subtract g(x_i) times the i-th column of A from the rhs
    rhs = -B.apply(R)

    # set g(x_i) for i-th dof in rhs
    bc_dofs = list(bc.get_boundary_values().keys())
    bc_vals = np.array([b[bc_dofs] for b in boundary_data])
    # FIXME currently, I have to use a workaround since
    # I don't know how to modify FenicsVectorArray in-place
    rhs_array = rhs.to_numpy()
    assert bc_vals.shape == (len(rhs), len(bc_dofs))
    rhs_array[:, bc_dofs] = bc_vals
    rhs = space.from_numpy(rhs_array)

    U = A.apply_inverse(rhs)
    return U


def extend(
    problem,
    boundary_data,
    solver_options={"solver": "mumps"},
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
    lhs_form = problem.get_form_lhs()
    df.assemble(lhs_form, tensor=A)

    # rhs
    B = A.copy()

    # prepare operator
    V = problem.V
    dummy = df.Function(V)
    bc = df.DirichletBC(V, dummy, df.DomainBoundary())
    bc.zero_columns(A, dummy.vector(), 1.0)

    # solver
    method = solver_options.get("solver")
    preconditioner = solver_options.get("preconditioner")
    if method == "lu" or method in df.lu_solver_methods():
        method = "default" if method == "lu" else method
        solver = df.LUSolver(A, method)
    else:
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
