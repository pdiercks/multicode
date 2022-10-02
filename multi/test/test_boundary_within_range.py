import dolfinx
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from multi.boundary import within_range


def test():
    n = 200
    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.quadrilateral
    )
    V = dolfinx.fem.FunctionSpace(domain, ("CG", 1))

    Δx = Δy = 1 / (
        n + 1
    )  # exclude the right and top boundary, Δ must be smaller than cell size
    boundary = within_range([0.0, 0.0, 0.0], [1.0 - Δx, 1.0 - Δy, 0.0])

    facet_dim = 1
    boundary_facets = dolfinx.mesh.locate_entities_boundary(domain, facet_dim, boundary)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, facet_dim, boundary_facets)
    bc = dolfinx.fem.dirichletbc(ScalarType(666), boundary_dofs, V)
    ndofs = bc.dof_indices()[1]
    # n_boundary_dofs = (n-1) * 4 + 4
    expected = (n - 1) * 2 + 1
    assert ndofs == expected


if __name__ == "__main__":
    test()
