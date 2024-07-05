import tempfile

import pytest
import numpy as np

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.io import gmshio
from basix.ufl import element
import ufl

from multi.boundary import plane_at
from multi.domain import RectangularDomain
from multi.problems import LinearProblem
from multi.extension import extend


def get_domain(name):
    if name == "rectangle":
        from multi.preprocessing import create_rectangle as create_mesh
    elif name == "voided_rectangle":
        from multi.preprocessing import create_voided_rectangle as create_mesh
    else:
        raise NotImplementedError
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_mesh(0., 1., 0., 1., num_cells=10, out_file=tf.name)
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_SELF, gdim=2)
    return domain


@pytest.mark.parametrize("name",["rectangle", "voided_rectangle"])
def test(name):
    domain = get_domain(name)
    quad = element("Lagrange", domain.basix_cell(), 2, shape=())
    V = fem.functionspace(domain, quad)
    Vdim = V.dofmap.index_map.size_global * V.dofmap.bs
    print(f"Number of DoFs={Vdim}")

    class DummyProblem(LinearProblem):
        def __init__(self, domain, V):
            super().__init__(domain, V)

        @property
        def form_lhs(self):
            u = self.trial
            v = self.test
            return ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

        @property
        def form_rhs(self):
            v = self.test
            f = fem.Constant(self.V.mesh, default_scalar_type(0.0))
            return ufl.inner(f, v) * ufl.dx

    def boundary_expression_factory(k):
        def expr(x):
            return x[0] * k

        return expr

    Ω = RectangularDomain(domain)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = np.array([], dtype=np.intc)
    for edge in Ω.boundaries:
        marker = Ω.str_to_marker(edge)
        entities = mesh.locate_entities_boundary(domain, fdim, marker)
        boundary_facets = np.append(boundary_facets, entities)

    problem = DummyProblem(Ω, V)

    boundary_data = []
    num_test = 20
    for i in range(1, num_test + 1):
        problem.clear_bcs()
        g = fem.Function(V)
        g.interpolate(boundary_expression_factory(i))
        if i < 15:
            # add fem.DirichletBC
            problem.add_dirichlet_bc(g, boundary_facets, entity_dim=1)
            bcs = problem.bcs
            boundary_data.append(bcs.copy())
        else:
            # add dict
            bc = {"value": g, "boundary": boundary_facets, "entity_dim": 1, "method": "topological"}
            boundary_data.append(list([bc]))

    # add another extension as list of dict
    zero = fem.Constant(domain, default_scalar_type(0.))
    one = fem.Constant(domain, default_scalar_type(1.))
    bottom = plane_at(0., "y")
    right = plane_at(1., "x")
    top = plane_at(1., "y")
    left = plane_at(0., "x")
    bcs = []
    bcs.append({"value": zero, "boundary": bottom, "method": "geometrical"})
    bcs.append({"value": zero, "boundary": right, "method": "geometrical"})
    bcs.append({"value": zero, "boundary": left, "method": "geometrical"})
    bcs.append({"value": one, "boundary": top, "method": "geometrical"})
    boundary_data.append(bcs)

    # compute extensions
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    extensions = extend(problem, boundary_facets, boundary_data, petsc_options=petsc_options)
    assert len(extensions) == num_test + 1
    assert all([not np.allclose(extensions[0].array, vec.array) for vec in extensions[1:]])

    def check_solution(j):
        problem.clear_bcs()
        if j == 21:
            for bc in bcs:
                problem.add_dirichlet_bc(**bc)
        else:
            g = fem.Function(V)
            g.interpolate(boundary_expression_factory(j))
            problem.add_dirichlet_bc(g, boundary_facets, entity_dim=1)
        uex = problem.solve()
        assert np.allclose(uex.vector[:], extensions[j - 1][:])

    for i in range(1, num_test + 2):
        check_solution(i)


if __name__ == "__main__":
    test()
