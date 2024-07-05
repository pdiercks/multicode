from typing import Union, Optional
import numpy as np
from dolfinx import fem
from petsc4py.PETSc import Vec as PETScVec
from multi.problems import LinearProblem


def extend(
    problem: LinearProblem,
    boundary_entities: np.ndarray,
    boundary_data: list[list[Union[fem.DirichletBC, dict]]],
    petsc_options: Optional[dict] = None,
) -> list[PETScVec]:
    """Extends the `boundary_data` into the domain of the `problem`.

    Args:
        problem: The linear problem, i.e. extension problem.
        boundary_entities: The entities associated with the Dirichlet boundary.
        boundary_data: The functions to be extended into the domain.
        See `multi.bcs.BoundaryConditions.add_dirichlet_bc` for `dict` values.
        petsc_options: PETSc options.

    """
    problem.clear_bcs()

    V = problem.V
    domain = V.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1

    # ### Assemble operator A
    zero_fun = fem.Function(V)
    zero_fun.x.petsc_vec.zeroEntries()
    problem.add_dirichlet_bc(
        zero_fun, boundary_entities, method="topological", entity_dim=fdim
    )
    problem.setup_solver(petsc_options=petsc_options)
    problem.assemble_matrix(bcs=problem.bcs)

    # define all extensions that should be computed
    assert all(
        [
            isinstance(bc, Union[fem.DirichletBC, dict])
            for bcs in boundary_data
            for bc in bcs
        ]
    )

    # ### initialize
    u = problem.u
    rhs = problem.b
    solver = problem.solver
    extensions = []

    for bcs in boundary_data:
        problem.clear_bcs()
        for bc in bcs:
            if isinstance(bc, dict):
                problem.add_dirichlet_bc(**bc)
            else:
                problem.add_dirichlet_bc(bc)
        current_bcs = problem.bcs

        problem.assemble_vector(current_bcs)
        solver.solve(rhs, u.x.petsc_vec)
        extensions.append(u.x.petsc_vec.copy())

    return extensions
