from mpi4py import MPI
import pytest
from dolfinx import fem, mesh
from basix.ufl import element
from multi.materials import LinearElasticMaterial


@pytest.mark.parametrize("gdim", list(range(3)))
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
        domain = mesh.create_unit_square(MPI.COMM_WORLD, 1, 1, 1, mesh.CellType.hexahedron)

    fe = element("P", domain.basix_cell(), 1, shape=(gdim,))
    V = fem.functionspace(domain, fe)


    if gdim == 1:
        u = fem.Function(V)
        u.interpolate(lambda x: (x[0],))

        breakpoint()
        strain = mat.eps(u)
        # TODO get integration points from basix
        # TODO strain = fem.Expression(mat.eps(u), points)

if __name__ == "__main__":
    test(1)
