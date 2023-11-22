from dolfinx import fem
from dolfinx import mesh

def extend(problem, boundary_data, petsc_options={}):
    """extend the `boundary_data` into the domain of the `problem`

    Parameters
    ----------
    problem : multi.LinearProblem
        The linear problem.
    boundary_data : list of list of dolfinx.fem.dirichletbc
        The boundary data to be extended.
    petsc_options : optional
        The petsc options for the linear problem.

    """
    problem.clear_bcs()

    V = problem.V
    domain = V.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1

    # add dummy bc on whole boundary
    # to zero out rows and columns of matrix A
    zero_fun = fem.Function(V)
    zero_fun.vector.zeroEntries()
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    problem.add_dirichlet_bc(
        zero_fun, boundary_facets, method="topological", entity_dim=fdim
    )
    bcs = problem.get_dirichlet_bcs()

    problem.setup_solver(petsc_options=petsc_options)
    solver = problem.solver

    problem.assemble_matrix(bcs)
    # clear bcs
    problem.clear_bcs()


    # define all extensions that should be computed
    assert all(
        [
            isinstance(bc, fem.DirichletBC)
            for bcs in boundary_data
            for bc in bcs
        ]
    )

    # initialize rhs vector
    rhs = problem.b
    # initialize solution
    u = fem.Function(problem.V)

    extensions = []
    for bcs in boundary_data:
        problem.clear_bcs()
        for bc in bcs:
            problem.add_dirichlet_bc(bc)
        current_bcs = problem.get_dirichlet_bcs()

        # set values to rhs
        problem.assemble_vector(current_bcs)
        solver.solve(rhs, u.vector)
        extensions.append(u.vector.copy())

    return extensions
