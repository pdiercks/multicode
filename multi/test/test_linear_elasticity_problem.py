import tempfile
import dolfinx
from dolfinx.io import gmshio
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from multi.boundary import plane_at
from multi.domain import Domain, RceDomain
from multi.preprocessing import create_rectangle_grid
from multi.problems import LinearElasticityProblem
from multi.misc import x_dofs_VectorFunctionSpace


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


def test_with_edges():
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(0., 1., 0., 1., num_cells=10, facets=True, recombine=True, out_file=tf.name)
        domain, ct, ft = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    Ω = RceDomain(domain, None, ft, edges=True)
    V = dolfinx.fem.VectorFunctionSpace(Ω.mesh, ("CG", 1))
    problem = LinearElasticityProblem(Ω, V, 210e3, 0.3)

    x_dofs = x_dofs_VectorFunctionSpace(problem.V)
    bottom = x_dofs[problem.V_to_L["bottom"]]
    assert np.allclose(bottom[:, 1], np.zeros_like(bottom[:, 1]))
    left = x_dofs[problem.V_to_L["left"]]
    assert np.allclose(left[:, 0], np.zeros_like(left[:, 0]))

    left = plane_at(0.0, "x")
    gdim = domain.geometry.dim
    zero = np.array((0, ) * gdim, dtype=PETSc.ScalarType)
    problem.add_dirichlet_bc(zero, boundary=left, method="geometrical")
    T = dolfinx.fem.Constant(domain, PETSc.ScalarType((1000., 0.)))
    problem.add_neumann_bc(3, T)

    u = problem.solve()

    Vdim = V.dofmap.bs * V.dofmap.index_map.size_global
    assert np.sum(problem._vector[:]) > 0.0
    assert problem._matrix[:, :].shape == (Vdim, Vdim)
    assert np.sum(np.abs(u.vector.array[:])) > 0.0


if __name__ == "__main__":
    test()
    test_with_edges()
