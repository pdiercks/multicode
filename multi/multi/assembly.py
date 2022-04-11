import numpy as np
import dolfin as df
from multi.bcs import compute_multiscale_bcs, apply_bcs
from pymor.bindings.fenics import FenicsMatrixOperator, FenicsVectorSpace
from pymor.tools.timing import Timer


def assemble_rom(
    logger,
    problem,
    dofmap,
    bases,
    edge_basis,
    cell_to_basis,
    dirichlet,
    edge_product,
    neumann,
):
    """build matrix and vector of ROM

    Parameters
    ----------
    logger : pymor.core.logging.getLogger
        An instance of the logger used in the main script.
    problem : multi.problems.LinearElasicityProblem
        The problem defined on the subdomain.
    dofmap : multi.dofmap.DofMap
        The DofMap of the reduced order model.
    bases : list of np.ndarray
        The reduced bases with local dof ordering.
    edge_basis : np.ndarray, optional
        The edge basis used to compute Dirichlet bcs.
        Must agree with corresponding bases.
    cell_to_basis : np.ndarray
        Mapping from cell index to basis (configuration).
    dirichlet : dict, optional
        Defines the Dirichlet BCs. Keys are cell indices and values are
        dict where keys are edge indices and values are dolfin.Expression
        or similar.
    edge_product : str, optional
        The inner product wrt which the edge basis was orthonormalized.
    neumann : dict, optional
        Defines the Neumann BCs. Keys are cell indices and values are
        list of dict specifying the Neumann data.

    Returns
    -------
    A : np.ndarray
        The left hand side.
    b: np.ndarray
        The right hand side.

    """

    # ### assemble local contributions once
    matrix = df.assemble(problem.get_form_lhs())
    operator = FenicsMatrixOperator(matrix, problem.V, problem.V)
    source = FenicsVectorSpace(problem.V)
    B = [source.from_numpy(rb) for rb in bases]
    A_local = [operator.apply2(basis, basis) for basis in B]

    # ### initialize global system
    N = dofmap.dofs()
    A = np.zeros((N, N))
    b = np.zeros(N)

    timer = Timer("rom")

    bcs = {}
    edge_to_str = "brtl"
    with logger.block("Start of Assembly loop ..."):
        timer.start()
        for cell_index, cell in enumerate(dofmap.cells):
            # translate since dirichlet bcs dependent on physical coord
            offset = np.around(dofmap.points[cell][0], decimals=3)  # lower left corner
            problem.domain.translate(df.Point(offset))
            dofs = dofmap.cell_dofs(cell_index)
            A[np.ix_(dofs, dofs)] += A_local[cell_to_basis[cell_index]]

            # ### Neumann BCs
            if cell_index in neumann.keys():
                neumann_bcs = neumann[cell_index]
                for force in neumann_bcs:
                    problem.add_neumann_bc(**force)
                F = df.assemble(problem.get_form_rhs())
                b_local = bases[cell_to_basis[cell_index]] @ F[:]
                b[dofs] += b_local

            # ### Dirichlet BCs
            if cell_index in dirichlet.keys():
                for edge, boundary_data in dirichlet[cell_index].items():
                    try:
                        # edge basis might be np.npzfile
                        chi = edge_basis[edge_to_str[edge]]
                    except IndexError:
                        # or np.ndarray in case of hierarchical basis
                        chi = edge_basis
                    bc_local = compute_multiscale_bcs(
                        problem,
                        cell_index,
                        edge,
                        boundary_data,
                        dofmap,
                        chi,
                        product=edge_product,
                        orth=True,
                    )
                    for k, v in bc_local.items():
                        bcs.update({k: v})

            # translate back
            problem.domain.translate(df.Point(-offset))
        timer.stop()
        logger.info(f"... Assembly took {timer.dt}s.")

    logger.info("Applying BCs to global ROM system ...")
    apply_bcs(A, b, list(bcs.keys()), list(bcs.values()))
    return A, b
