import numpy as np
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import Function
from multi.bcs import BoundaryDataFactory
from multi.extension import extend
from multi.shapes import NumpyQuad


def compute_phi(problem, nodes):
    """compute coarse scale basis functions for given problem

    Note:
        This assumes a rectangular domain.
    """
    V = problem.V
    quadrilateral = NumpyQuad(nodes)
    shape_functions = quadrilateral.interpolate(V)

    boundary_entities = np.array([], dtype=np.intc)
    for edge in problem.domain.boundaries:
        marker = problem.domain.str_to_marker(edge)
        entities = locate_entities_boundary(problem.domain.grid, problem.domain.tdim-1, marker)
        boundary_entities = np.append(boundary_entities, entities)

    data_factory = BoundaryDataFactory(problem.domain.grid, boundary_entities, V)

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
    phi = extend(problem, boundary_entities, boundary_data, petsc_options=petsc_options)
    return phi
