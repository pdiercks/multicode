import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from multi.domain import Domain
from multi.problems import LinearProblem
from multi.extension import extend


def test():
    num_cells = int(100 / 1.41)
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, num_cells, num_cells, dolfinx.mesh.CellType.quadrilateral
    )
    V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2))
    Vdim = V.dofmap.index_map.size_global * V.dofmap.bs
    print(f"Number of DoFs={Vdim}")

    class DummyProblem(LinearProblem):
        def __init__(self, domain, V):
            super().__init__(domain, V)

        @property
        def form_lhs(self):
            u = self.u
            v = self.v
            return ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

        @property
        def form_rhs(self):
            v = self.v
            f = dolfinx.fem.Constant(self.V.mesh, ScalarType(0.0))
            return f * v * ufl.dx

    def bottom(x):
        return np.isclose(x[1], 0.0)

    def boundary_expression_factory(k):
        def expr(x):
            return x[0] * k

        return expr

    Ω = Domain(domain)
    problem = DummyProblem(Ω, V)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)

    boundary_data = []
    num_test = 20
    for i in range(1, num_test + 1):
        problem.clear_bcs()
        g = dolfinx.fem.Function(V)
        g.interpolate(boundary_expression_factory(i))
        problem.add_dirichlet_bc(g, boundary_facets, entity_dim=1)
        bcs = problem.get_dirichlet_bcs()
        boundary_data.append(bcs.copy())

    # compute extensions
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    extensions = extend(problem, boundary_data, petsc_options=petsc_options)
    assert len(extensions) == num_test
    for fun in extensions:
        print(np.sum(fun[:]))
    # assert np.allclose(extensions[1][:], extensions[7][:])

    # check one of the solutions
    j = 3
    problem.clear_bcs()
    g = dolfinx.fem.Function(V)
    g.interpolate(boundary_expression_factory(j))
    problem.add_dirichlet_bc(g, boundary_facets, entity_dim=1)
    uex = problem.solve()

    assert np.allclose(uex.vector[:], extensions[j - 1][:])


if __name__ == "__main__":
    test()
