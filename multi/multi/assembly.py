import dolfinx
import numpy as np
from multi.bcs import compute_multiscale_bcs
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from pymor.tools.timing import Timer


def assemble_rom(
    logger,
    problem,
    dofmap,
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
    assert len(bases) == dofmap.num_cells

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

    grid = dofmap.grid

    timer = Timer("rom")
    bcs = {}
    with logger.block("Start of Assembly loop ..."):
        timer.start()
        for cell_index in range(dofmap.num_cells):
            # translate since dirichlet bcs dependent on physical coord
            vertices = grid.get_entities(0, cell_index)
            dx = dolfinx.mesh.compute_midpoints(grid.mesh, 0, vertices[0])
            dx = np.around(dx, decimals=4)
            problem.domain.translate(dx)

            # FIXME careful with translation ...
            # It might happen that later on when trying to interpolate
            # function on boundary points of the domain, these points
            # are actually not found ...
            # currently this is solved by using np.around inside
            # multi.bcs.compute_multiscale_bcs when computing vertex coordinates
            # TODO? use StructuredQuadGrid.create_fine_grids to "instantiate"
            # RCE grid for each coarse grid cell ? --> then no translation is needed
            # at all, but have to read mesh and build function space etc. each time

            dofs = dofmap.cell_dofs(cell_index)
            A[np.ix_(dofs, dofs)] += A_local[cell_index]

            # ### Neumann BCs
            if cell_index in neumann.keys():
                neumann_bcs = neumann[cell_index]
                for marker, value in neumann_bcs:
                    problem.add_neumann_bc(marker, value)
                F = problem.assemble_vector()
                b_local = bases[cell_index] @ F.array
                b[dofs] += b_local

            # ### Dirichlet BCs
            if project_dirichlet:
                if cell_index in dirichlet.keys():
                    for edge, boundary_data in dirichlet[cell_index].items():
                        if edge_basis is not None:
                            try:
                                # edge basis might be np.npzfile
                                chi = edge_basis[edge]
                            except IndexError:
                                # or np.ndarray in case of hierarchical basis
                                chi = edge_basis
                        else:
                            chi = None
                        # FIXME specific to block problem
                        # or do this always like this??
                        g = dolfinx.fem.Function(problem.V)
                        g.interpolate(boundary_data)
                        bc_local = compute_multiscale_bcs(
                            problem,
                            cell_index,
                            edge,
                            g,
                            dofmap,
                            chi,
                            product=edge_product,
                            orth=True,
                        )
                        for k, v in bc_local.items():
                            bcs.update({k: v})
                        if cell_index == 10:
                            breakpoint()

            # translate back
            problem.domain.translate(-dx)
        timer.stop()
        logger.info(f"... Assembly took {timer.dt}s.")

    if not project_dirichlet:
        bcs.update(dirichlet)

    return A, b, bcs
