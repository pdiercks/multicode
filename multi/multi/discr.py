"""discretization module"""

import numpy as np
import dolfin as df

from multi import Domain, LinearElasticityProblem
from multi.shapes import NumpyQuad
from pymor.bindings.fenics import (
    FenicsVectorSpace,
    FenicsMatrixOperator,
    FenicsVisualizer,
)
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.parameters.functionals import ProjectionParameterFunctional


def discretize_rve(rvemeshfile, degree, material, translate=None, opts=None):
    """discretize given rve domain and wrap as pyMOR model
    where the rhs is given as a linear combination of
    lagrange shape functions of order 1 and given parameter
    as coefficients

    Parameters
    ----------
    rvemeshfile : Path, str
        The RVE mesh.
    degree : int
        Degree of the lagrange space.
    material : dict
        Material metadata.
    translate : optional
        How to translate the RVE domain in space.
    opts : optional
        Solver options.

    Returns
    -------
    fom : The pymor instance of the Full order model.
    problem: The linear elasticity problem.
    """
    rve_domain = Domain(
        rvemeshfile,
        id_=0,
        subdomains=True,
        edges=True,
        translate=translate,
    )
    E = material["Material parameters"]["E"]["value"]
    NU = material["Material parameters"]["NU"]["value"]

    V = df.VectorFunctionSpace(rve_domain.mesh, "CG", degree)
    problem = LinearElasticityProblem(
        rve_domain, V, E=E, NU=NU, plane_stress=material["Constraints"]["plane_stress"]
    )
    a = problem.get_lhs()
    A = df.assemble(a)

    # boundary data g
    q = df.Function(V)
    g = df.Function(V)

    def boundary(x, on_boundary):
        return on_boundary

    quad_nodes = np.array(
        [
            [rve_domain.xmin, rve_domain.ymin],
            [rve_domain.xmax, rve_domain.ymin],
            [rve_domain.xmax, rve_domain.ymax],
            [rve_domain.xmin, rve_domain.ymax],
        ]
    )
    quadrilateral = NumpyQuad(quad_nodes)

    vector_operators = []
    null = np.zeros(V.dim())
    S = FenicsVectorSpace(V)
    shapes = quadrilateral.interpolate(V)
    for shape in shapes:
        g.vector().set_local(null)
        A_bc = A.copy()
        q.vector().set_local(shape)
        bc = df.DirichletBC(V, q, boundary)
        bc.zero_columns(A_bc, g.vector(), 1.0)
        vector_operators.append(VectorOperator(S.make_array([g.vector().copy()])))

    parameter_functionals = [
        ProjectionParameterFunctional("mu", shapes.shape[0], index=i)
        for i in range(shapes.shape[0])
    ]

    rhs = LincombOperator(vector_operators, parameter_functionals)
    if opts is not None:
        solver_options = {
            "inverse": {
                "solver": opts["solver_parameters"]["linear_solver"],
                "preconditioner": opts["solver_parameters"]["preconditioner"],
            }
        }
    else:
        solver_options = None
    lhs = FenicsMatrixOperator(A_bc, V, V, solver_options=solver_options)

    # ### products
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
        error_estimator=None,
        visualizer=FenicsVisualizer(S),
        name="RVE",
    )
    return fom, problem


def discretize_block(
    problem, gamma, serendipity=True, additional_bcs=(), forces=(), solver=None
):
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
    forces : tuple of dict, optional
        Neumann bcs on boundaries other than `gamma` given as tuple of dict,
        where dict matches the signature of `multi.bcs.MechanicsBCs.add_force`.
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

    # ### rhs
    S = FenicsVectorSpace(V)
    vector_operators = []
    parameter_functionals = []

    # add any sources
    n_forces = len(forces)
    if n_forces > 0:
        for k, force in enumerate(forces):
            problem.bc_handler.add_force(**force)
        L = problem.get_rhs()
        b = df.assemble(L)
        vector_operators.append(VectorOperator(S.make_array([b])))
        parameter_functionals.append(1.0)

    # add bc for each of the shape functions
    R = []
    for k, shape in enumerate(shapes):
        r = df.Function(V, name="r_" + str(k))
        R.append(r)
        shape_function = df.Function(V)
        shape_function.vector().set_local(shape)
        problem.bc_handler.add_bc(gamma, shape_function)
        parameter_functionals.append(
            ProjectionParameterFunctional("mu", shapes.shape[0], index=k)
        )

    # add additional dirichlet bcs
    n_add_bcs = len(additional_bcs)
    if n_add_bcs > 0:
        for j, abc in enumerate(additional_bcs):
            R.append(df.Function(V, name="r_" + str(k + j)))
            problem.bc_handler.add_bc(**abc)
            parameter_functionals.append(1.0)

    bcs = problem.bc_handler.bcs()
    n_bcs = len(bcs)
    a = problem.get_lhs()
    A = df.assemble(a)

    # add vector operators for all inhomogeneous dirichlet bcs
    for i in range(n_bcs):
        bcs[i].zero_columns(A.copy(), R[i].vector(), 1.0)
        vector_operators.append(VectorOperator(S.make_array([R[i].vector().copy()])))

    # ### operator
    # lift bcs for one of the shapes and for each additional bc
    A_0 = A.copy()
    dummy = df.Function(V)
    bcs[0].zero_columns(A_0, dummy.vector(), 1.0)
    for j in range(n_add_bcs):
        bcs[len(shapes) + j].zero_columns(A_0, dummy.vector(), 1.0)

    if solver is not None:
        solver_options = {
            "inverse": {
                "solver": solver["solver_parameters"]["linear_solver"],
                "preconditioner": solver["solver_parameters"]["preconditioner"],
            }
        }
    else:
        solver_options = None

    rhs = LincombOperator(vector_operators, parameter_functionals)
    lhs = FenicsMatrixOperator(A_0, V, V, solver_options=solver_options)

    fom = StationaryModel(
        lhs,
        rhs,
        output_functional=None,
        products={
            "energy": FenicsMatrixOperator(A.copy(), V, V, name="energy"),
            "energy_0": FenicsMatrixOperator(A_0, V, V, name="energy_0"),
        },
        error_estimator=None,
        visualizer=FenicsVisualizer(S),
        name="BLOCK",
    )
    return fom
