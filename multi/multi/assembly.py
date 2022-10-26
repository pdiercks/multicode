import dolfinx
import numpy as np
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from pymor.core.logger import getLogger
from pymor.tools.timing import Timer
from multi.interpolation import interpolate

"""Design

inputs:
    multiscale problem --> info global BC etc.
    subdomain_problem --> compute projected operators
    dofmap
    bases

returns:
    A, b, bcs

    determine bc dofs and values (depends on kind of dirichlet bc)
------------------------------------------------------------------
    (a) f = const. --> evaluate f at vertices, get fine scale modes dofs and set to 0.
    (there should not be any modes, because f=const --> triggers f=0 in oversampling)
    (b) f = non zero --> evaluate f at vertices, get fine scale modes and set to 1. since
    these should be correctly computed already
    (c) f = 0 --> evaluate f at vertices?

    ... f might not be a Function?
    ... how does the dofmap determine the dofs? --> geometrically via locator?


• the multiscale problem should define a cell set 'neumann' or 'dirichlet'
• therefore I know which cells are on said boundaries
• using the StructuredQuadGrid, the entities of the boundary can be determined by marker
• thus still need multiscale_problem.get_dirichlet(ci)["boundary"]

first check for inhomogeneous dirichlet,
--> if yes --> apply 
--> if None --> check if homogeneous ones and apply
"""


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
    subdomain_problem.clear_bcs()  # do not apply bcs to matrix
    subdomain_problem.setup_solver()  # create matrix object
    matrix = subdomain_problem.assemble_matrix()
    operator = FenicsxMatrixOperator(matrix, subdomain_problem.V, subdomain_problem.V)
    source = FenicsxVectorSpace(subdomain_problem.V)
    B = [source.from_numpy(rb) for rb in bases]
    A_local = [operator.apply2(basis, basis) for basis in B]

    # ### initialize global system
    grid = dofmap.grid
    N = dofmap.num_dofs
    A = np.zeros((N, N))
    b = np.zeros(N)

    timer = Timer("assembly")
    with logger.block("Start of Assembly loop ..."):
        timer.start()
        for cell_index in range(dofmap.num_cells):

            dofs = dofmap.cell_dofs(cell_index)
            A[np.ix_(dofs, dofs)] += A_local[cell_index]

            if cell_index in neumann_set:
                # assemble F and project
                # this has to be done locally (on subdomain level)
                raise NotImplementedError

    logger.info("Computing Dirichlet BCs for ROM")
    dirichlet = multiscale_problem.get_dirichlet()
    bcs = {}
    for bc in dirichlet["inhomogeneous"]:
        locator = bc["boundary"]
        uD = bc["value"]

        # locate entities
        vertices = grid.locate_entities_boundary(0, locator)
        edges = grid.locate_entities_boundary(1, locator)
        # entity coordinates
        x_verts = grid.get_entity_coordinates(0, vertices)

        # FIXME this interpolation does produce unwanted result
        # if bc is component-wise, but inhomogeneous

        # coarse scale dofs
        coarse_values = interpolate(uD, x_verts)
        for values, ent in zip(coarse_values, vertices):
            dofs = dofmap.entity_dofs(0, ent)
            for k, v in zip(values, dofs):
                bcs.update({k: v})

        # fine scale dofs
        for ent in edges:
            dofs = dofmap.entity_dofs(1, ent)
            assert len(dofs) == 1
            bcs.update({dofs[0]: 1.})

    for bc in dirichlet["homogeneous"]:
        locator = bc["boundary"]
        uD = bc["value"]

        # locate entities
        vertices = grid.locate_entities_boundary(0, locator)
        edges = grid.locate_entities_boundary(1, locator)

        # this sets zero to all dofs
        # wrong for component-wise homogeneous bc
        for ent in vertices:
            dofs = dofmap.entity_dofs(0, ent)
            for d in dofs:
                bcs.update({d: 0.0})

        # no need to deal with the edges
        # elif bc_type == "homogeneous":
        #     for ent in edges:
        #         dofs = dofmap.entity_dofs(1, ent)
        #         if len(dofs) < 1:
        #             # there are no modes for this edge
        #             # do nothing
        #         else:
        #             # component-wise homogeneous bc
        #             # do nothing

        # TODO what about a constant function?


        # TODO edges
        # inhom --> there should be a single mode --> v=1.
        # hom --> either no mode because fully constrained and hence no reason to train this in the oversampling
        # OR component-wise hom bc --> several modes should be present , but no dof values should be set ... --> free component dof values are determined from system solve

        # NOTE 26.10.2022
        # I get the feeling that this won't work
        # I need more information here, than what is defined in BlockProblem (MultiscaleProblem) on a global level
        # It would have been nice to define the dirichlet conditions once in a certain form and be able to use that
        # everywhere, but I guess it just does not work ...

    return A, b, bcs
