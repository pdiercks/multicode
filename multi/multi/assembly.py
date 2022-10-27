import numpy as np
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from pymor.core.logger import getLogger
from pymor.tools.timing import Timer
from multi.bcs import compute_dirichlet_online


def assemble_rom(
    multiscale_problem,
    subdomain_problem,
    dofmap,
    bases
):
    """assemble matrix and vector of ROM and compute bcs

    Parameters
    ----------
    multiscale_problem : multi.problem.MultiscaleProblem
        The global problem defining cell sets, BCs etc.
    subdomain_problem : multi.problems.LinearProblem
        The problem defined on the subdomain.
    dofmap : multi.dofmap.DofMap
        The DofMap of the reduced order model.
    bases : list of np.ndarray
        The reduced bases with local dof ordering.

    Returns
    -------
    A : np.ndarray
        The left hand side.
    b : np.ndarray
        The right hand side.
    bcs : dict
        The dof indices (keys) and values (values).

    """
    logger = getLogger("multi.assembly.assemble_rom")
    assert len(bases) == dofmap.num_cells

    # ### assemble local contributions once
    timer = Timer("assembly")
    timer.start()
    subdomain_problem.clear_bcs()  # do not apply bcs to matrix
    subdomain_problem.setup_solver()  # create matrix object
    matrix = subdomain_problem.assemble_matrix()
    operator = FenicsxMatrixOperator(matrix, subdomain_problem.V, subdomain_problem.V)
    source = FenicsxVectorSpace(subdomain_problem.V)
    B = [source.from_numpy(rb) for rb in bases]
    A_local = [operator.apply2(basis, basis) for basis in B]
    timer.stop()
    logger.info(f"Assembled local operators in {timer.dt}s.")

    # ### initialize global system
    N = dofmap.num_dofs
    A = np.zeros((N, N))
    b = np.zeros(N)

    cell_sets = multiscale_problem.cell_sets

    with logger.block("Start of Assembly loop ..."):
        timer.start()
        for cell_index in range(dofmap.num_cells):

            dofs = dofmap.cell_dofs(cell_index)
            A[np.ix_(dofs, dofs)] += A_local[cell_index]

            if cell_index in cell_sets["neumann"]:
                # assemble F and project
                # this has to be done locally (on subdomain level)
                # b += b_local
                raise NotImplementedError
        timer.stop()
    logger.info(f"Assembled ROM system in {timer.dt}s.")

    dirichlet = multiscale_problem.get_dirichlet()
    timer.start()
    bcs = compute_dirichlet_online(dofmap, dirichlet)
    timer.stop()
    logger.info(f"Computed Dirichlet bcs in {timer.dt}s.")
    return A, b, bcs
