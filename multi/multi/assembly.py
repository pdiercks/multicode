import numpy as np
from multi.bcs import compute_multiscale_bcs, apply_bcs
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from pymor.tools.timing import Timer


def assemble_rom(
    logger,
    problem,
    dofmap,
    cell_to_basis,
    bases,
    edge_basis=None,
    dirichlet=None,
    edge_product="h1",
    neumann=None,
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
    cell_to_basis : np.ndarray
        Mapping from cell index to basis (configuration).
    bases : list of np.ndarray
        The reduced bases with local dof ordering.
    edge_basis : np.ndarray or np.lib.npyio.NpzFile, optional
        The edge basis used to compute Dirichlet bcs.
        Must agree with corresponding bases.
    dirichlet : dict, optional
        Defines the Dirichlet BCs.
        (a) Keys are bc_dofs and values are bc_values.
        (b) Keys are cell indices and values are dict where keys are edge
        indices and values are functions (dolfinx.fem.Function).
        (b) can be used to compute multiscale bcs via projection and should
        contain extra key value pair ('compute_multiscale_bcs', True).
    edge_product : str, optional
        The inner product wrt which the edge basis was orthonormalized.
    neumann : dict, optional
        Defines the Neumann BCs. Keys are cell indices and values are
        list of tuple (marker, value) specifying the Neumann data.

    Returns
    -------
    A : np.ndarray
        The left hand side.
    b: np.ndarray
        The right hand side.

    """

    # ### assemble local contributions once
    problem.clear_bcs()  # do not apply bcs to matrix
    problem.setup_solver()  # create matrix object
    matrix = problem.assemble_matrix()
    operator = FenicsxMatrixOperator(matrix, problem.V, problem.V)
    source = FenicsxVectorSpace(problem.V)
    B = [source.from_numpy(rb) for rb in bases]
    A_local = [operator.apply2(basis, basis) for basis in B]

    # ### initialize global system
    N = dofmap.num_dofs()
    A = np.zeros((N, N))
    b = np.zeros(N)

    dirichlet = dirichlet or {}
    project_dirichlet = dirichlet.pop("compute_multiscale_bcs", False)
    neumann = neumann or {}

    timer = Timer("rom")
    bcs = {}
    edge_to_str = "brtl"
    with logger.block("Start of Assembly loop ..."):
        timer.start()
        for cell_index, cell in enumerate(dofmap.cells):
            # translate since dirichlet bcs dependent on physical coord
            offset = np.around(dofmap.points[cell][0], decimals=3)  # lower left corner
            problem.domain.translate(offset)
            dofs = dofmap.cell_dofs(cell_index)
            A[np.ix_(dofs, dofs)] += A_local[cell_to_basis[cell_index]]

            # ### Neumann BCs
            if cell_index in neumann.keys():
                neumann_bcs = neumann[cell_index]
                # TODO double check neumann bcs input is correctly defined
                # Neumann bc should be tuple of (marker, values)
                # marker (int in range(4)+1) should be local to the respective RceDomain
                for marker, value in neumann_bcs:
                    problem.add_neumann_bc(marker, value)
                F = problem.assemble_vector()
                # F should be PETSc.Vec
                b_local = bases[cell_to_basis[cell_index]] @ F.array
                b[dofs] += b_local

            # ### Dirichlet BCs
            if project_dirichlet:
                assert edge_basis is not None
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
            problem.domain.translate(-offset)
        timer.stop()
        logger.info(f"... Assembly took {timer.dt}s.")

    logger.info("Applying BCs to global ROM system ...")
    if not project_dirichlet:
        bcs.update(dirichlet)
    apply_bcs(A, b, list(bcs.keys()), list(bcs.values()))
    return A, b
