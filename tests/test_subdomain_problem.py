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


@pytest.mark.parametrize("create_grid",[create_rectangle,create_voided_rectangle])
def test(create_grid):
    ul = 1000.0
    width = ul
    height = ul
    num_cells = 16

    args = (0.0, width, 0.0, height)

    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        kwargs = {"num_cells": num_cells, "recombine": True, "facet_tags": {"bottom": 1, "left": 2, "right": 3, "top": 4}, "out_file": tf.name}
        if create_grid.__name__ == 'create_voided_rectangle':
            kwargs.update({"radius": 0.2 * ul})
        create_grid(*args, **kwargs)
        domain, _, ft = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    ve = element("Lagrange", domain.basix_cell(), 1, shape=(2,))
    V = fem.functionspace(domain, ve)
    gdim = domain.geometry.dim
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
    problem.create_edge_space_maps()

    # check mappings between spaces

    # FIXME correct the mappings
    # TODO use new function of SubProblem for that

    x_dofs = x_dofs_vectorspace(problem.V)
    bottom = x_dofs[problem.V_to_L["bottom"]]
    top = x_dofs[problem.V_to_L["top"]]
    top_to_bottom = problem.edge_space_maps["top_to_bottom"]
    assert np.allclose(bottom[:, 1], np.zeros_like(bottom[:, 1]))
    assert np.allclose(top[:, 1], height * np.ones_like(top[:, 1]))
    assert np.allclose(bottom[:, 0], top[top_to_bottom, 0])

    left = x_dofs[problem.V_to_L["left"]]
    right = x_dofs[problem.V_to_L["right"]]
    right_to_left = problem.edge_space_maps["right_to_left"]
    assert np.allclose(left[:, 0], np.zeros_like(left[:, 0]))
    assert np.allclose(right[:, 0], width * np.ones_like(right[:, 0]))
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

    bcs = problem.bcs
    problem.assemble_matrix(bcs)
    problem.assemble_vector(bcs)

    u = problem.u
    solver.solve(problem.b, u.x.petsc_vec)
    u.x.scatter_forward()

    Vdim = V.dofmap.bs * V.dofmap.index_map.size_global
    assert np.sum(problem.b[:]) > 0.0
    assert problem.A[:, :].shape == (Vdim, Vdim)
    assert np.sum(np.abs(u.x.array[:])) > 0.0
