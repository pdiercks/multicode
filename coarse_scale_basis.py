import dolfin as df
import numpy as np
from multi.shapes import NumpyQuad
from pymor.bindings.fenics import (
    FenicsVectorSpace,
    FenicsMatrixOperator,
    FenicsVisualizer,
)
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.models.basic import StationaryModel
from pymor.parameters.functionals import ProjectionParameterFunctional


def construct_coarse_scale_basis(problem, solver_options=None, return_fom=False):
    """construct coarse scale basis for given subdomain problem

    Parameters
    ----------
    problem : a suitable problem like :class:`~multi.linear_elasticity.LinearElasticityProblem`
    solver_options : dict, optional
        A dict of strings. See https://github.com/pymor/pymor/blob/2021.2.0/src/pymor/operators/interface.py
    return_fom : bool, optional
        If True, the full order model is returned.

    Returns
    -------
    basis : FenicsVectorArray
        The coarse scale basis for given subdomain.
    fom
        The full order model (if `return_fom` is `True`).

    """
    # full system matrix
    A = df.assemble(problem.get_lhs())

    # ### define bilinear shape functions as inhomogeneous dirichlet bcs
    V = problem.V
    f = df.Function(V)  # placeholder for rhs vector operators
    g = df.Function(V)  # boundary data
    n_vertices = 4  # assumes rectangular domain
    nodes = problem.domain.get_nodes(n=n_vertices)
    quadrilateral = NumpyQuad(nodes)
    shape_functions = quadrilateral.interpolate(V)

    null = np.zeros(V.dim())
    space = FenicsVectorSpace(V)
    vector_operators = []
    for shape in shape_functions:
        f.vector().set_local(null)
        g.vector().set_local(shape)
        bc = df.DirichletBC(V, g, df.DomainBoundary())
        A_bc = A.copy()
        bc.zero_columns(A_bc, f.vector(), 1.0)
        vector_operators.append(VectorOperator(space.make_array([f.vector().copy()])))

    lhs = FenicsMatrixOperator(A_bc, V, V, solver_options=solver_options)
    parameter_functionals = [
        ProjectionParameterFunctional("mu", shape_functions.shape[0], index=i)
        for i in range(shape_functions.shape[0])
    ]
    rhs = LincombOperator(vector_operators, parameter_functionals)

    # ### inner products
    energy_mat = A.copy()
    energy_0_mat = A_bc.copy()
    l2_mat = problem.get_product(name="l2", bcs=False)
    l2_0_mat = l2_mat.copy()
    h1_mat = problem.get_product(name="h1", bcs=False)
    h1_0_mat = h1_mat.copy()
    bc.apply(l2_0_mat)
    bc.apply(h1_0_mat)

    fom = StationaryModel(
        lhs,
        rhs,
        output_functional=None,
        products={
            "energy": FenicsMatrixOperator(energy_mat, V, V, name="energy"),
            "energy_0": FenicsMatrixOperator(energy_0_mat, V, V, name="energy_0"),
            "l2": FenicsMatrixOperator(l2_mat, V, V, name="l2"),
            "l2_0": FenicsMatrixOperator(l2_0_mat, V, V, name="l2_0"),
            "h1": FenicsMatrixOperator(h1_mat, V, V, name="h1"),
            "h1_0": FenicsMatrixOperator(h1_0_mat, V, V, name="h1_0"),
        },
        estimator=None,
        visualizer=FenicsVisualizer(space),
        name="FOM",
    )

    # ### compute the coarse scale basis
    dim = 2  # spatial dimension
    z = dim * n_vertices
    basis = fom.operator.source.empty(reserve=z)
    Identity = np.eye(z)
    for row in Identity:
        basis.append(fom.solve({"mu": row}))

    if return_fom:
        return basis, fom
    else:
        return basis
