import tempfile
import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmshio
from basix.ufl import element
from multi.boundary import plane_at
from multi.domain import RectangularSubdomain
from multi.materials import LinearElasticMaterial
from multi.misc import x_dofs_vectorspace
from multi.preprocessing import create_rectangle, create_voided_rectangle
from multi.problems import LinElaSubProblem
from multi.interpolation import make_mapping


@pytest.mark.parametrize("create_grid",[create_rectangle,create_voided_rectangle])
def test(create_grid):
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_grid(
            0.0,
            1.0,
            0.0,
            1.0,
            num_cells=10,
            facets=True,
            recombine=True,
            out_file=tf.name,
        )
        domain, _, ft = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    ve = element("Lagrange", domain.basix_cell(), 1, shape=(2,))
    V = fem.functionspace(domain, ve)
    gdim = domain.ufl_cell().geometric_dimension()
    phases = LinearElasticMaterial(gdim, 210e3, 0.3, plane_stress=True)
    Ω = RectangularSubdomain(1, domain, facet_tags=ft)
    problem = LinElaSubProblem(Ω, V, phases)

    with pytest.raises(AttributeError):
        Ω.create_boundary_grids()

    with pytest.raises(AttributeError):
        # edge meshes are not yet setup for Ω
        problem.setup_edge_spaces()

    with pytest.raises(AttributeError):
        # coarse grid not yet setup
        problem.setup_coarse_space()

    Ω.create_coarse_grid(1)
    Ω.create_boundary_grids()
    problem.create_map_from_V_to_L() # sets up edge spaces if required
    problem.setup_coarse_space()

    # check mappings between spaces
    x_dofs = x_dofs_vectorspace(problem.V)
    bottom = x_dofs[problem.V_to_L["bottom"]]
    top = x_dofs[problem.V_to_L["top"]]
    top_to_bottom = make_mapping(problem.edge_spaces["fine"]["bottom"], problem.edge_spaces["fine"]["top"])
    assert np.allclose(bottom[:, 1], np.zeros_like(bottom[:, 1]))
    assert np.allclose(top[:, 1], np.ones_like(top[:, 1]))
    assert np.allclose(bottom[:, 0], top[top_to_bottom, 0])

    left = x_dofs[problem.V_to_L["left"]]
    right = x_dofs[problem.V_to_L["right"]]
    right_to_left = make_mapping(problem.edge_spaces["fine"]["left"], problem.edge_spaces["fine"]["right"])
    assert np.allclose(left[:, 0], np.zeros_like(left[:, 0]))
    assert np.allclose(right[:, 0], np.ones_like(right[:, 0]))
    assert np.allclose(left[:, 1], right[right_to_left, 1])

    # arbitrary solution of the problem
    left = plane_at(0.0, "x")
    gdim = domain.geometry.dim
    zero = np.array((0,) * gdim, dtype=default_scalar_type)
    problem.add_dirichlet_bc(zero, boundary=left, method="geometrical")
    T = fem.Constant(domain, default_scalar_type((1000.0, 0.0)))
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

    u = fem.Function(V)
    solver.solve(problem.b, u.vector)
    u.x.scatter_forward()

    Vdim = V.dofmap.bs * V.dofmap.index_map.size_global
    assert np.sum(problem.b[:]) > 0.0
    assert problem.A[:, :].shape == (Vdim, Vdim)
    assert np.sum(np.abs(u.vector.array[:])) > 0.0


if __name__ == "__main__":
    test()
