from mpi4py import MPI
import pytest
from dolfinx import fem, mesh, default_scalar_type
import basix
from basix.ufl import element
import numpy as np
from multi.materials import LinearElasticMaterial


@pytest.mark.parametrize("gdim", [1, 2, 3])
def test(gdim):

    if gdim == 1:
        cell_type = basix.CellType.interval
        domain = mesh.create_unit_interval(MPI.COMM_WORLD, 1)
    elif gdim == 2:
        cell_type = basix.CellType.quadrilateral
        domain = mesh.create_unit_square(MPI.COMM_WORLD, 1, 1, mesh.CellType.quadrilateral)
    else:
        cell_type = basix.CellType.hexahedron
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, mesh.CellType.hexahedron)

    E = fem.Constant(domain, default_scalar_type(210e3)) # 210 GPa
    NU = fem.Constant(domain, default_scalar_type(0.3))
    mat = LinearElasticMaterial(gdim, E=E, NU=NU)
    fe = element("P", domain.basix_cell(), 1, shape=(gdim,))
    V = fem.functionspace(domain, fe)

    map_c = domain.topology.index_map(domain.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    ipoint, _ = basix.make_quadrature(cell_type, 1)
    ix = ipoint[0, 0]

    u = fem.Function(V)
    alpha = 12.4

    def u_data_factory(gdim):

        def u(x):
            zero = np.zeros_like(x[0])
            ux = x[0] ** 2 + alpha * x[0]
            uy = x[0] + x[1]
            uz = zero
            if gdim == 1:
                return (ux, )
            elif gdim == 2:
                return (ux, uy)
            elif gdim == 3:
                return (ux, uy, uz)

        return u

    u.interpolate(u_data_factory(gdim))

    def strain_sum(gdim):
        # define analytic solution for each gdim
        if gdim == 1:
            return 2. * ix + alpha
        elif gdim > 1:
            return 2. * ix + alpha + 2.

    strain_expr = fem.Expression(mat.eps(u), ipoint)
    strain_vals = strain_expr.eval(domain, cells)
    assert strain_vals.size == 9
    assert np.isclose(np.sum(strain_vals), strain_sum(gdim))


if __name__ == "__main__":
    test(3)
