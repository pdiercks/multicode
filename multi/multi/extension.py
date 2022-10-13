import dolfinx

# NOTE consider this to set values to FenicsxVectorArray
# (a) via DirichletBC
# from petsc4py import PETSc
# dolfinx.fem.petsc.set_bc(U.vectors[0].real_part.impl, [bc])
# U.vectors[0].real_part.impl.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
# (b) via .array instance of the petsc.vector
# U.vectors[1].real_part.impl.array[:] = np.arange(source.dim)
# (c) via petsc.vector.setArray()
# U.vectors[2].real_part.impl.setArray(np.linspace(0, 1, num=source.dim))

# Design
# apply_lifting: pymor vs. fenicsx?

# fenicsx:
#     lhs = dolfinx.form(ufl_form_lhs)
#     rhs = dolfinx.form(ufl_form_rhs)
#     matrix = dolfinx.fem.petsc.create_matrix(lhs)
#     vector = dolfinx.fem.petsc.create_vector(rhs)

#     # assembly
#     matrix.zeroEntries()
#     dolfinx.fem.petsc.assemble_matrix(matrix, lhs, bcs=bcs)
#     matrix.assemble()

#     vector.zeroEntries()
#     dolfnix.fem.petsc.assemble_vector(vector, rhs)
#     vector.ghostUpdate(
#         addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
#     )

#     # Compute b - J(u_D-u_(i-1))
#     dolfinx.fem.petsc.apply_lifting(vector, [lhs], [bcs])
#     # Set dx|_bc = u_{i-1}-u_D
#     dolfinx.fem.petsc.set_bc(vector, bcs, scale=1.0)
#     self._vector.ghostUpdate(
#         addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
#     )
# --> only need to provide an array of list of bc that represent function to extend
# --> solve linear problem by looping over array of bcs
# pros: only assemble matrix once provided bc is associated with whole boundary
# contra: not vectorized

# pymor version
# assemble matrix without bc --> compute rhs (apply lifting)
# assemble matrix with bc
# loop over bc to set_bc for rhs (same loop over array of bcs as in the fenicsx version)

"""pure fenicsx version

domain = ...
V = ...
problem = LinearProblem(domain, V, solver_options)

solver = problem.setup_solver()

# assemble matrix once before the loop; can use dummy bc with all boundary dofs
zero_fun = dolfinx.fem.Function(problem.V)
zero_fun.vector.zeroEntries()
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
problem.add_dirichlet_bc(zero_fun, boundary_facets, method='topological', entity_dim=1)
matrix = problem.assemble_matrix()
problem.clear_bcs()

# define all extensions that should be computed
boundary_data = [[bc00, bc01], [bc1], [bc2]]

extensions = []
for bcs in boundary_data:
    problem.clear_bcs()
    for bc in bcs:
        problem.add_dirichlet_bc(bc)
    rhs_vector = problem.assemble_vector()
    f = dolfinx.fem.Function(V)
    solver.solve(rhs_vector, f.vector)
    extensions.append(f.vector)

VA = source.make_array(extensions)

"""


def extend(problem, boundary_data):
    """extend the `boundary_data` into the domain of the `problem`

    Parameters
    ----------
    problem : multi.LinearProblem
        The linear problem.
    boundary_data : list of list of dolfinx.fem.dirichletbc
        The boundary data to be extended.
    """
    problem.clear_bcs()

    V = problem.V
    domain = V.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1

    solver = problem.setup_solver()

    # assemble matrix once before the loop
    zero_fun = dolfinx.fem.Function(V)
    zero_fun.vector.zeroEntries()
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    problem.add_dirichlet_bc(
            zero_fun, boundary_facets, method='topological', entity_dim=fdim
            )
    bcs = problem.get_dirichlet_bcs()
    problem.assemble_matrix(bcs)
    problem.clear_bcs()

    # define all extensions that should be computed
    assert all([isinstance(bc, dolfinx.fem.DirichletBCMetaClass) for bcs in boundary_data for bc in bcs])

    extensions = []
    for bcs in boundary_data:
        problem.clear_bcs()
        for bc in bcs:
            problem.add_dirichlet_bc(bc)
        current_bcs = problem.get_dirichlet_bcs()
        rhs = problem.assemble_vector(current_bcs)
        f = dolfinx.fem.Function(V)
        fvec = f.vector
        solver.solve(rhs, fvec)
        extensions.append(fvec.copy())

    return extensions
