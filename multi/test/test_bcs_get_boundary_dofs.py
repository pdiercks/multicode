import dolfinx
from mpi4py import MPI
from multi.bcs import BoundaryConditions


def test():
    n = 8
    degree = 2
    dim = 2

    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, dolfinx.mesh.CellType.quadrilateral
    )
    V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", degree), dim=dim)

    bc_handler = BoundaryConditions(domain, V)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    u = dolfinx.fem.Function(V)
    u.x.set(0.)
    bc_handler.add_dirichlet_bc(u, boundary_facets, method="topological", entity_dim=1)
    bcs = bc_handler.bcs
    dofs = bcs[0].dof_indices()[0]
    assert dofs.size == n * 4 * degree * dim
    # there are (n+1) * 4 - 4 points
    # and if degree == 2: additional n*4 dofs (edges)
    # thus n * 4 * degree dofs for degree in (1, 2)
    # times number of components (i.e. dim)


if __name__ == "__main__":
    test()
