import dolfin as df
from multi.product import InnerProduct
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenics import FenicsVectorSpace


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
    options = solver_options or _solver_options()
    method = options.get("solver")
    preconditioner = options.get("preconditioner")
    if method == "lu" or method in df.lu_solver_methods():
        method = "default" if method == "lu" else method
        solver = df.LUSolver(matrix, method)
    else:
        solver = df.KrylovSolver(matrix, method, preconditioner)
    return solver


def build_nullspace2D(V, u):
    """Function to build null space for 2D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [u.copy() for i in range(3)]
    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0)

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[2], -1.0, 1)
    V.sub(1).set_x(nullspace_basis[2], 1.0, 0)

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = df.VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis


def build_nullspace_va(func_space, product=None):
    f = df.Function(func_space)
    fvec = f.vector()

    # Create list of vectors for null space
    nullspace_basis = [fvec.copy() for i in range(3)]
    # Build translational null space basis
    func_space.sub(0).dofmap().set(nullspace_basis[0], 1.0)
    func_space.sub(1).dofmap().set(nullspace_basis[1], 1.0)

    # Build rotational null space basis
    func_space.sub(0).set_x(nullspace_basis[2], -1.0, 1)
    func_space.sub(1).set_x(nullspace_basis[2], 1.0, 0)

    for x in nullspace_basis:
        x.apply("insert")

    source = FenicsVectorSpace(func_space)
    basis = source.make_array(nullspace_basis)

    inner_product = InnerProduct(func_space, product=product)
    product = inner_product.assemble_operator()
    gram_schmidt(basis, product, atol=0.0, rtol=0.0, copy=False)

    return basis, product


def get_cg_solver(null_space, operator):
    """get cg solver"""

    df.as_backend_type(operator).set_near_nullspace(null_space)
    pc = df.PETScPreconditioner("petsc_amg")

    df.PETScOptions.set("mg_levels_ksp_type", "chebyshev")
    df.PETScOptions.set("mg_levels_pc_type", "jacobi")

    # Improve estimate of eigenvalues for Chebyshev smoothing
    df.PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
    df.PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

    lin_solver = df.PETScKrylovSolver("cg", pc)
    lin_solver.parameters["monitor_convergence"] = True
    lin_solver.parameters["nonzero_initial_guess"] = True
    lin_solver.parameters["maximum_iterations"] = 1000
    lin_solver.parameters["relative_tolerance"] = 1.0e-9
    lin_solver.parameters["absolute_tolerance"] = 1.0e-12
    lin_solver.parameters["error_on_nonconvergence"] = True
    lin_solver.set_operator(operator)
    return lin_solver
