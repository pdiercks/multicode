from dolfinx.fem import Function
from multi.bcs import BoundaryDataFactory
from multi.extension import extend
from multi.shapes import NumpyQuad


def compute_phi(problem, nodes):
    """compute coarse scale basis functions for given problem"""
    V = problem.V
    quadrilateral = NumpyQuad(nodes)
    shape_functions = quadrilateral.interpolate(V)

    data_factory = BoundaryDataFactory(problem.domain.grid, V)

    boundary_data = []
    g = Function(V)
    for shape in shape_functions:
        g.x.array[:] = shape
        boundary_data.append([data_factory.create_bc(g.copy())])

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    phi = extend(problem, boundary_data, petsc_options=petsc_options)
    return phi
