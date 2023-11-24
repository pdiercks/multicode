# Design for LinearProblem

```python
# init
p = LinearProblem(domain, V)

# set bcs unique to this problem
p.add_dirichlet_bc(...)
p.add_neumann_bc(...)

# make sure that p.get_form_lhs() and p.get_form_rhs() are implemented ...

# 3. setup solver
p.setup_solver(petsc_options, form_compiler_options, jit_options)

# ----------------
# now with the solver setup there are different use cases
# (A) solve the problem once and be done for today
p.solve() # will call super.solve()

# ----------------
# (B) we want to solve the same problem many times with different rhs
solver = p.solver # first get the solver we did setup
p.assemble_matrix(bcs) # assemble matrix once for specific set of bcs

# Note that p.A is filled with values internally
# Next, we only need to define the rhs for which we want to solve
# One option is to assemble the vector based on p.get_form_rhs()

p.assemble_vector(bcs) # let assemble vector always modify p.b
rhs = p.b
solution = dolfinx.fem.Function(p.V)
solver.solve(rhs, solution.vector)
solution.x.scatter_forward()

# Another option could be to create a function and set some values to it
rhs = dolfinx.fem.Function(p.V)
with rhs.vector.localForm() as rhs_loc:
    rhs_loc.set(0)
assemble_vector(rhs.vector, some_compiled_form) or skip this
apply_lifting(rhs.vector, [p.a], bcs=[p.get_dirichlet_bcs()])
rhs.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(rhs.vector, p.get_dirichlet_bcs())
solver.solve(rhs.vector, solution.vector)
solution.x.scatter_forward()
```
