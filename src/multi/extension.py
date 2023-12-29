from typing import Union, Optional
from multi.problems import LinearProblem
from dolfinx import fem, mesh
from petsc4py.PETSc import Vec as PETScVec


def extend(
    problem: LinearProblem,
    boundary_data: list[list[Union[fem.DirichletBC, dict]]],
    petsc_options: Optional[dict] = None,
) -> list[PETScVec]:
    """Extends the `boundary_data` into the domain of the `problem`.

    Args:
        problem: The linear problem, i.e. extension problem.
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
    zero_fun.vector.zeroEntries()
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    problem.add_dirichlet_bc(
        zero_fun, boundary_facets, method="topological", entity_dim=fdim
    )
    problem.setup_solver(petsc_options=petsc_options)
    problem.assemble_matrix(bcs=problem.get_dirichlet_bcs())

    # define all extensions that should be computed
    assert all(
        [
            isinstance(bc, Union[fem.DirichletBC, dict])
            for bcs in boundary_data
            for bc in bcs
        ]
    )

    # ### initialize
    u = fem.Function(V)
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
        current_bcs = problem.get_dirichlet_bcs()

        problem.assemble_vector(current_bcs)
        solver.solve(rhs, u.vector)
        extensions.append(u.vector.copy())

    return extensions
