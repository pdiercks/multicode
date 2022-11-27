import dolfinx


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
    zero_fun = dolfinx.fem.Function(V)
    zero_fun.vector.zeroEntries()
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
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
            isinstance(bc, dolfinx.fem.DirichletBCMetaClass)
            for bcs in boundary_data
            for bc in bcs
        ]
    )

    # initialize rhs vector
    rhs = problem.b
    # initialize solution
    u = dolfinx.fem.Function(problem.V)

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


# FIXME dim or boundary, but not both are required?
# TODO can use interpolation between different meshes now
# maybe this is not needed anymore
def restrict(function, marker, dim, boundary=False):
    """restrict the function to some part of the domain

    Parameters
    ----------
    function : dolfinx.fem.Function
        The function to be evaluated at the boundary.
    marker : callable
        A function that defines the subdomain geometrically.
        `dolfinx.mesh.locate_entities` is used.
    dim : int
        Topological dimension of the entities.
    boundary : optional, bool
        If True, use `dolfinx.mesh.locate_entities_boundary` and
        locate entities on the boundary only.

    Returns
    -------
    function values restricted to some part of the domain
    """

    V = function.function_space
    domain = V.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    assert dim in (fdim, tdim)

    if boundary:
        entities = dolfinx.mesh.locate_entities_boundary(domain, fdim, marker)
    else:
        entities = dolfinx.mesh.locate_entities(domain, tdim, marker)

    dofs = dolfinx.fem.locate_dofs_topological(V, dim, entities)
    dummy = dolfinx.fem.Function(V)
    dummy.x.set(0.0)
    bc = dolfinx.fem.dirichletbc(dummy, dofs)
    dof_indices = bc.dof_indices()[0]
    return function.vector.array[dof_indices]
