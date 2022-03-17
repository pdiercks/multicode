import numpy as np
import dolfin as df
from multi.extension import extend
from multi import Domain, LinearElasticityProblem
from multi.basis_construction import construct_coarse_scale_basis
from multi.shapes import NumpyQuad

# Test for PETSc
if not df.has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

# Set backend to PETSC
prm = df.parameters
prm["linear_algebra_backend"] = "PETSc"
solver_prm = prm["krylov_solver"]
solver_prm["relative_tolerance"] = 1e-9
solver_prm["absolute_tolerance"] = 1e-12


def test():
    mesh = df.UnitSquareMesh(10, 10)
    V = df.VectorFunctionSpace(mesh, "CG", 2)
    domain = Domain(mesh)
    problem = LinearElasticityProblem(domain, V, E=60e3, NU=0.2, plane_stress=True)
    solver_options = {"solver": "cg", "preconditioner": "default"}
    phi = construct_coarse_scale_basis(problem, solver_options=solver_options)
    space = phi.space

    # alternatively, construct coarse scale basis via extension
    vertices = domain.get_nodes(n=4)
    quadrilateral = NumpyQuad(vertices)
    shapes = quadrilateral.interpolate(V)
    boundary_data = []
    for shape in shapes:
        g = df.Function(V)
        gvec = g.vector()
        gvec.zero()
        gvec.set_local(shape)
        boundary_data.append(g)

    for i in range(len(shapes)):
        ed = shapes[i] - boundary_data[0].vector()[:]
        assert np.sum(ed) < 1e-9

    coarse = extend(problem, boundary_data, solver_options=solver_options)
    U = space.make_array(coarse)
    err = phi - U
    norm = err.norm()  # should be near relative tolerance ...
    assert np.all(norm < 1e-7)
    test = np.allclose(phi.to_numpy(), U.to_numpy())
    assert test


if __name__ == "__main__":
    test()
