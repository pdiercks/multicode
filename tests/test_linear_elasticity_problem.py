import tempfile

import numpy as np

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.io import gmshio
from basix.ufl import element

from multi.boundary import plane_at
from multi.domain import Domain, RectangularSubdomain
from multi.materials import LinearElasticMaterial
from multi.preprocessing import create_unit_cell_01
from multi.problems import LinearElasticityProblem
from multi.interpolation import interpolate


def test():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    # helpers
    left = plane_at(0.0, "x")
    right = plane_at(1.0, "x")

    # create domain with facet markers for neumann bc
    marker_value = 17
    right_boundary_facets = mesh.locate_entities_boundary(domain, fdim, right)
    facet_tags = mesh.meshtags(
        domain, fdim, right_boundary_facets, marker_value
    )
    Ω = Domain(domain, facet_tags=facet_tags)

    # initialize problem
    fe = element("P", domain.basix_cell(), 1, shape=(2,))
    V = fem.functionspace(domain, fe)
    gdim = domain.geometry.dim

    youngs_mod = fem.Constant(domain, default_scalar_type(210e3))
    poisson_ratio = fem.Constant(domain, default_scalar_type(0.3))
    phases = LinearElasticMaterial(gdim, E=youngs_mod, NU=poisson_ratio, plane_stress=True)
    problem = LinearElasticityProblem(Ω, V, phases)

    # add dirichlet and neumann bc
    zero = fem.Constant(domain, (default_scalar_type(0.0),) * 2)
    problem.add_dirichlet_bc(zero, left, method="geometrical")
    f_ext = fem.Constant(
        domain, (default_scalar_type(1000.0), default_scalar_type(0.0))
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

    u = fem.Function(V)
    rhs = problem.b
    solver.solve(rhs, u.vector)

    assert np.isclose(np.sum(rhs[:]), 1000.0)
    assert np.sum(np.abs(u.x.array[:])) > 0.0

    # high level solve
    other = problem.solve()
    assert np.allclose(other.vector.array[:], u.vector.array[:])

    # update material and solve problem again after re-assembly
    new_material = ({"E": 105e3, "NU": 0.3},)
    problem.update_material(new_material)
    problem.assemble_matrix(bcs)
    solver.solve(rhs, u.vector)
    assert not np.allclose(other.vector.array[:], u.vector.array[:])
    assert np.isclose(other.x.array.max() * 2, u.x.array.max())


def test_dirichlet():
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_unit_cell_01(
            0.0,
            1.0,
            0.0,
            1.0,
            num_cells=8,
            cell_tags={"matrix": 1, "inclusion": 2},
            out_file=tf.name,
        )
        domain, ct, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    Ω = RectangularSubdomain(1, domain, cell_tags=ct)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(tdim, fdim)

    # helpers
    left = plane_at(0.0, "x")
    right = plane_at(1.0, "x")
    bottom = plane_at(0.0, "y")

    # initialize problem
    fe = element("P", domain.basix_cell(), 1, shape=(2,))
    V = fem.functionspace(domain, fe)
    gdim = domain.geometry.dim
    phases = [
        (LinearElasticMaterial(gdim, 210e3, 0.3, plane_stress=True), 1),
        (LinearElasticMaterial(gdim, 210e3, 0.3, plane_stress=True), 2),
    ]
    problem = LinearElasticityProblem(Ω, V, phases)

    # add two dirichlet bc
    zero = fem.Constant(domain, (default_scalar_type(0.0),) * 2)
    f_x = 12.3
    f_y = 0.0
    f = fem.Constant(domain, (default_scalar_type(f_x), default_scalar_type(f_y)))

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

    u = fem.Function(V)
    solver.solve(problem.b, u.vector)
    u.x.scatter_forward()

    assert np.sum(problem.b[:]) > 0
    assert np.sum(np.abs(u.x.array[:])) > 0.0

    # extract values at the bottom
    bdofs = fem.locate_dofs_geometrical(V, bottom)
    xdofs = V.tabulate_dof_coordinates()
    x_bottom = xdofs[bdofs]
    u_values = interpolate(u, x_bottom)
    assert np.isclose(
        np.sum(f_x * np.linspace(0, 1, num=9, endpoint=True)), np.sum(u_values)
    )

    # extract value at the right
    bdofs = fem.locate_dofs_geometrical(V, right)
    xdofs = V.tabulate_dof_coordinates()
    x_right = xdofs[bdofs]
    u_values = interpolate(u, x_right)
    assert np.isclose(np.sum(u_values), f_x * 9)


if __name__ == "__main__":
    test()
    test_dirichlet()
