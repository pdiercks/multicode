"""discretization module"""

import numpy as np
import dolfin as df

from multi.shapes import NumpyQuad
from pymor.bindings.fenics import (
    FenicsVectorSpace,
    FenicsMatrixOperator,
    FenicsVisualizer,
)
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.parameters.functionals import ProjectionParameterFunctional


def discretize_block(problem, gamma, serendipity=True, additional_bcs=(), solver=None):
    """pyMOR discretization of a oversampling problem

    Parameters
    ----------
    problem : `multi.LinearElasticityProblem`
        The linear problem defined on the oversampling domain.
    gamma : boundary
        The boundary where to apply load parametrization. See
        `multi.bcs.MechanicsBCs.add_bc` for possible values of `boundary`.
    serendipity : bool, optional
        If True, use serendipity shape functions for load parametrization.
    additional_bcs : tuple of dict, optional
        BCs on boundaries other than `gamma` given as tuple of dict, where
        dict matches the signature of `multi.bcs.MechanicsBCs.add_bc`.
    solver : optional
        Solver options given by `multi.misc.get_solver`.

    Returns
    -------
    fom : pymor.models.basic.StationaryModel
        The full order model for the oversampling domain.

    """
    domain = problem.domain
    V = problem.V

    xmin = domain.xmin
    xmax = domain.xmax
    ymin = domain.ymin
    ymax = domain.ymax

    def mid_point(a, b):
        return a + (b - a) / 2

    points = [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
        [mid_point(xmin, xmax), ymin],
        [xmax, mid_point(ymin, ymax)],
        [mid_point(xmin, xmax), ymax],
        [xmin, mid_point(ymin, ymax)],
    ]
    if not serendipity:
        points.append([mid_point(xmin, xmax), mid_point(ymin, ymax)])

    quad = NumpyQuad(np.array(points))
    shapes = quad.interpolate(V)

    # rhs functions
    R = []
    for k, shape in enumerate(shapes):
        R.append(df.Function(V, name="r_" + str(k)))
        shape_function = df.Function(V)
        shape_function.vector().set_local(shape)
        problem.bc_handler.add_bc(gamma, shape_function)

    n_add_bcs = len(additional_bcs)
    if n_add_bcs > 0:
        for j, abc in enumerate(additional_bcs):
            R.append(df.Function(V, name="r_" + str(k + j)))
            problem.bc_handler.add_bc(**abc)

    bcs = problem.bc_handler.bcs()
    n_bcs = len(bcs)
    a = problem.get_lhs()
    A = df.assemble(a)

    # ### rhs vector operators
    S = FenicsVectorSpace(V)
    vop = []
    for i in range(n_bcs):
        bcs[i].zero_columns(A.copy(), R[i].vector(), 1.0)
        vop.append(VectorOperator(S.make_array([R[i].vector().copy()])))

    parameter_functionals = [
        ProjectionParameterFunctional("mu", shapes.shape[0], index=i)
        for i in range(shapes.shape[0])
    ]

    # ### operator
    # lift bcs for one of the shapes and for each additional bc
    A_0 = A.copy()
    dummy = df.Function(V)
    bcs[0].zero_columns(A_0, dummy.vector(), 1.0)
    for j in range(n_add_bcs):
        bcs[len(shapes) + j].zero_columns(A_0, dummy.vector(), 1.0)
        # add 1.0 for each additional bc
        parameter_functionals.append(1.0)

    if solver is not None:
        solver_options = {
            "inverse": {
                "solver": solver["solver_parameters"]["linear_solver"],
                "preconditioner": solver["solver_parameters"]["preconditioner"],
            }
        }
    else:
        solver_options = None

    rhs = LincombOperator(vop, parameter_functionals)
    lhs = FenicsMatrixOperator(A_0, V, V, solver_options=solver_options)

    fom = StationaryModel(
        lhs,
        rhs,
        output_functional=None,
        products={
            "energy": FenicsMatrixOperator(A.copy(), V, V, name="energy"),
            "energy_0": FenicsMatrixOperator(A_0, V, V, name="energy_0"),
        },
        estimator=None,
        visualizer=FenicsVisualizer(S),
        name="BLOCK",
    )
    return fom
