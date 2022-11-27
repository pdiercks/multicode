import tempfile
import dolfinx
from dolfinx.io import gmshio
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from multi.boundary import plane_at
from multi.domain import Domain, RectangularDomain
from multi.preprocessing import create_rectangle_grid
from multi.problems import LinearElasticityProblem
from multi.misc import x_dofs_VectorFunctionSpace
from multi.interpolation import interpolate


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
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", 1))
    problem = LinearElasticityProblem(Ω, V, 210e3, 0.3)

    # import ufl
    # r = ufl.rank(problem.form_lhs)
    # print(r)
    # breakpoint()

    # add dirichlet and neumann bc
    zero = dolfinx.fem.Constant(domain, (PETSc.ScalarType(0.0),) * 2)
    problem.add_dirichlet_bc(zero, left, method="geometrical")
    f_ext = dolfinx.fem.Constant(
        domain, (PETSc.ScalarType(1000.0), PETSc.ScalarType(0.0))
    )
    problem.add_neumann_bc(marker_value, f_ext)


    # setup the solver
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem.setup_solver(petsc_options=petsc_options)
    solver = problem.solver

    bcs = problem.get_dirichlet_bcs()
    problem.assemble_matrix(bcs)
    problem.assemble_vector(bcs)

    u = dolfinx.fem.Function(V)
    rhs = problem.b
    solver.solve(rhs, u.vector)

    assert np.isclose(np.sum(rhs[:]), 1000.0)
    assert np.sum(np.abs(u.x.array[:])) > 0.0

    # high level solve
    other = problem.solve()
    assert np.allclose(other.vector.array[:], u.vector.array[:])


def test_dirichlet():
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    # helpers
    left = plane_at(0.0, "x")
    right = plane_at(1.0, "x")
    bottom = plane_at(0.0, "y")

    Ω = Domain(domain, cell_markers=None, facet_markers=None)

    # initialize problem
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", 1))
    problem = LinearElasticityProblem(Ω, V, 210e3, 0.3)

    # add two dirichlet bc
    zero = dolfinx.fem.Constant(domain, (PETSc.ScalarType(0.0),) * 2)
    f_x = 12.3
    f_y = 0.0
    f = dolfinx.fem.Constant(domain, (PETSc.ScalarType(f_x), PETSc.ScalarType(f_y)))

    def u_bottom(x):
        return (x[0] * f_x, x[1])

    problem.add_dirichlet_bc(zero, left, method="geometrical")
    problem.add_dirichlet_bc(f, right, method="geometrical")
    problem.add_dirichlet_bc(u_bottom, bottom, method="geometrical")

    petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            }
    problem.setup_solver(petsc_options=petsc_options)
    solver = problem.solver

    bcs = problem.get_dirichlet_bcs()
    problem.assemble_matrix(bcs)
    problem.assemble_vector(bcs)

    u = dolfinx.fem.Function(V)
    solver.solve(problem.b, u.vector)
    u.x.scatter_forward()

    assert np.sum(problem.b[:]) > 0
    assert np.sum(np.abs(u.x.array[:])) > 0.0

    # extract values at the bottom
    bdofs = dolfinx.fem.locate_dofs_geometrical(V, bottom)
    xdofs = V.tabulate_dof_coordinates()
    x_bottom = xdofs[bdofs]
    u_values = interpolate(u, x_bottom)
    assert np.isclose(np.sum(f_x * np.linspace(0, 1, num=9, endpoint=True)), np.sum(u_values))

    # extract value at the right
    bdofs = dolfinx.fem.locate_dofs_geometrical(V, right)
    xdofs = V.tabulate_dof_coordinates()
    x_right = xdofs[bdofs]
    u_values = interpolate(u, x_right)
    assert np.isclose(np.sum(u_values), f_x * 9)


def test_with_edges():
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(
            0.0,
            1.0,
            0.0,
            1.0,
            num_cells=10,
            facets=True,
            recombine=True,
            out_file=tf.name,
        )
        domain, ct, ft = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    Ω = RectangularDomain(domain, None, ft)
    Ω.create_edge_grids(10)
    V = dolfinx.fem.VectorFunctionSpace(Ω.grid, ("Lagrange", 1))

    problem = LinearElasticityProblem(Ω, V, 210e3, 0.3)
    problem.setup_edge_spaces()
    problem.create_map_from_V_to_L()

    x_dofs = x_dofs_VectorFunctionSpace(problem.V)
    bottom = x_dofs[problem.V_to_L["bottom"]]
    assert np.allclose(bottom[:, 1], np.zeros_like(bottom[:, 1]))
    left = x_dofs[problem.V_to_L["left"]]
    assert np.allclose(left[:, 0], np.zeros_like(left[:, 0]))

    left = plane_at(0.0, "x")
    gdim = domain.geometry.dim
    zero = np.array((0,) * gdim, dtype=PETSc.ScalarType)
    problem.add_dirichlet_bc(zero, boundary=left, method="geometrical")
    T = dolfinx.fem.Constant(domain, PETSc.ScalarType((1000.0, 0.0)))
    problem.add_neumann_bc(3, T)

    petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            }
    problem.setup_solver(petsc_options=petsc_options)
    solver = problem.solver

    bcs = problem.get_dirichlet_bcs()
    problem.assemble_matrix(bcs)
    problem.assemble_vector(bcs)

    u = dolfinx.fem.Function(V)
    solver.solve(problem.b, u.vector)
    u.x.scatter_forward()

    Vdim = V.dofmap.bs * V.dofmap.index_map.size_global
    assert np.sum(problem.b[:]) > 0.0
    assert problem.A[:, :].shape == (Vdim, Vdim)
    assert np.sum(np.abs(u.vector.array[:])) > 0.0


if __name__ == "__main__":
    test()
    test_dirichlet()
    test_with_edges()
