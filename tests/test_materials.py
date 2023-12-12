from mpi4py import MPI
import pytest
from dolfinx import fem, mesh
import basix
from basix.ufl import element
import numpy as np
from multi.materials import LinearElasticMaterial


@pytest.mark.parametrize("gdim", [1, 2, 3])
def test(gdim):
    E = 210e3 # 210 GPa
    NU = 0.3

    mat = LinearElasticMaterial(gdim, E=E, NU=NU)
    if gdim == 1:
        domain = mesh.create_unit_interval(MPI.COMM_WORLD, 1)
    elif gdim == 2:
        domain = mesh.create_unit_square(MPI.COMM_WORLD, 1, 1, mesh.CellType.quadrilateral)
        mat_edz = LinearElasticMaterial(gdim, E=E, NU=NU, plane_stress=False)
        mat_esz = LinearElasticMaterial(gdim, E=E, NU=NU, plane_stress=True)
    else:
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, mesh.CellType.hexahedron)

    fe = element("P", domain.basix_cell(), 1, shape=(gdim,))
    V = fem.functionspace(domain, fe)

    map_c = domain.topology.index_map(domain.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    if gdim == 1:
        u = fem.Function(V)
        magic_number = 12.4
        u.interpolate(lambda x: (x[0] ** 2 + magic_number * x[0],))

        points, _ = basix.make_quadrature(basix.CellType.interval, 1)
        strain_expr = fem.Expression(mat.eps(u), points)
        strain_vals = strain_expr.eval(domain, cells)
        assert strain_vals.size == 9
        assert np.isclose(np.sum(strain_vals), 1. + magic_number)


if __name__ == "__main__":
    test(1)
