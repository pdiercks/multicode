import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from multi.boundary import plane_at
from multi.domain import Domain
from multi.problems import LinearElasticityProblem


def test():
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    # helpers
    left = plane_at(0.0, "x")
    right = plane_at(1.0, "x")

    # create domain with facet markers for neumann bc
    marker_value = 17
    right_boundary_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, right)
    facet_tags = dolfinx.mesh.meshtags(
        domain, fdim, right_boundary_facets, marker_value
    )
    Ω = Domain(domain, cell_markers=None, facet_markers=facet_tags)

    # initialize problem
    V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
    problem = LinearElasticityProblem(Ω, V, 210e3, 0.3)

    # add dirichlet and neumann bc
    zero = dolfinx.fem.Constant(domain, (PETSc.ScalarType(0.0),) * 2)
    problem.add_dirichlet_bc(zero, left, method="geometrical")
    f_ext = dolfinx.fem.Constant(
        domain, (PETSc.ScalarType(1000.0), PETSc.ScalarType(0.0))
    )
    problem.add_neumann_bc(marker_value, f_ext)

    u = problem.solve()
    vector = problem._vector
    matrix = problem._matrix

    assert np.isclose(np.sum(vector[:]), 1000.0)
    Vdim = V.dofmap.index_map.size_global * V.dofmap.bs
    assert matrix[:, :].shape == (Vdim, Vdim)
    assert np.sum(np.abs(u.x.array[:])) > 0.0


# def test_with_edges():
#     domain = RectangularDomain(
#         "data/rcedomain.xdmf", _id=1, subdomains=False, edges=True
#     )
#     V = df.VectorFunctionSpace(domain.mesh, "CG", 1)
#     problem = LinearElasticityProblem(domain, V, 210e3, 0.3)

#     x_dofs = problem.V.tabulate_dof_coordinates()
#     bottom = x_dofs[problem.V_to_L[0]]
#     assert np.allclose(bottom[:, 1], np.zeros_like(bottom[:, 1]))
#     left = x_dofs[problem.V_to_L[3]]
#     assert np.allclose(left[:, 0], np.zeros_like(left[:, 0]))

#     left = plane_at(0.0)
#     right = plane_at(1.0)
#     problem.add_dirichlet_bc(left, df.Constant((0, 0)))
#     problem.add_neumann_bc(right, df.Constant((1000, 0)))
#     a = problem.get_form_lhs()
#     L = problem.get_form_rhs()

#     u = df.Function(V)
#     df.solve(a == L, u, problem.dirichlet_bcs())

#     assert np.sum(df.assemble(L)[:]) > 0.0
#     assert df.assemble(a).array().shape == (V.dim(), V.dim())
#     assert np.sum(np.abs(u.vector()[:])) > 0.0


if __name__ == "__main__":
    test()
    # test_with_edges()
