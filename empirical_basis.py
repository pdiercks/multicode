"""
construction of empirical basis
(1) phi_i: bilinear functions which also account for the
    inheterogenity
(2) psi_j: edge functions which also account for the
    inheterogenity
Note: edge functions are computed via POD of edge snapshot data.
Optionally the first edge functions can be defined as hierarchical
polynomials up to a maximum degree pmax (see Options below).

Usage:
    empirical_basis.py [options] BLOCK RVE A DEG MAT

Arguments:
    BLOCK               Filepath for block grid.
    RVE                 XDMF file path (incl. extension).
    A                   The unit length of the RVE.
    DEG                 Degree of (fine grid) FE space.
    MAT                 The material parameters (.yml).

Options:
    -h, --help               Show this message.
    -l LEVEL, --log=LEVEL    Set the log level [default: 30].
    --solver=SOLVER          Define dolfin solver parameters.
    --training-set=SET       Provide a training set for the block problem. This can
                             either be a '.npy' file or a keyword ('random', 'delta').
                             [default: random].
    --pmax=PMAX              Use hierarchical polynomials up to pmax to initialize
                             set of edge functions [default: 1].
    -o FILE, --output=FILE   Specify output path for empirical basis (.npy).
    --plot-errors            Plot projection errors.
    --projerr=FILE           Write projection errors to given path (.txt).
    --chi=FILE               Write edge basis functions to path (.npy).
    --chi-svals=FILE         Write edge basis singular values to path (.npy).
    --rtol=RTOL              Relative tolerance to be used with POD [default: 4e-8].
    --check-interface=TOL    Check interface compatibility with tolerance TOL.
    --test                   Run tests to verify implementation.
"""

import sys
from pathlib import Path
from time import time

import yaml
import dolfin as df
import numpy as np
from docopt import docopt
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# multi
from multi import Domain, LinearElasticityProblem, make_mapping
from multi.shapes import NumpyQuad

# pymor
from pymor.algorithms.pod import pod
from pymor.bindings.fenics import (
    FenicsMatrixOperator,
    FenicsVectorSpace,
    FenicsVisualizer,
)
from pymor.core.exceptions import AccuracyError
from pymor.core.logger import getLogger
from pymor.models.basic import StationaryModel
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.basic import extend_basis


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["BLOCK"] = Path(args["BLOCK"])
    args["RVE"] = Path(args["RVE"])
    args["A"] = float(args["A"])
    args["DEG"] = int(args["DEG"])
    with open(Path(args["MAT"]), "r") as infile:
        try:
            args["material"] = yaml.safe_load(infile)
        except yaml.YAMLError as exc:
            print(exc)
    args["--pmax"] = int(args["--pmax"])
    args["--rtol"] = float(args["--rtol"])
    args["--log"] = int(args["--log"])
    args["--output"] = Path(args["--output"]) if args["--output"] else None
    args["--chi"] = Path(args["--chi"]) if args["--chi"] else None
    args["--chi-svals"] = Path(args["--chi-svals"]) if args["--chi-svals"] else None
    args["--check-interface"] = (
        float(args["--check-interface"]) if args["--check-interface"] else None
    )
    if args["--training-set"] not in ("random", "delta"):
        args["--training-set"] = Path(args["--training-set"])
        if not args["--training-set"].exists():
            print(f"The training set {args['--training-set']} does not exist.")
            sys.exit(1)

    solver = {
        "krylov_solver": {
            "relative_tolerance": 1.0e-9,
            "absolute_tolerance": 1.0e-12,
            "maximum_iterations": 1000,
        },
        "solver_parameters": {"linear_solver": "default", "preconditioner": "default"},
    }
    if args["--solver"] is not None:
        assert Path(args["--solver"]).suffix == ".yml"
        try:
            with open(args["--solver"], "r") as f:
                solver = yaml.safe_load(f)
        except FileNotFoundError:
            print(
                f"File {args['--solver']} could not be found. Using default solver settings ..."
            )
    args["solver"] = solver
    prm = df.parameters
    prm["krylov_solver"]["relative_tolerance"] = solver["krylov_solver"][
        "relative_tolerance"
    ]
    prm["krylov_solver"]["absolute_tolerance"] = solver["krylov_solver"][
        "absolute_tolerance"
    ]
    prm["krylov_solver"]["maximum_iterations"] = solver["krylov_solver"][
        "maximum_iterations"
    ]
    return args


def compute_coarse_scale_basis(args, fom):
    training_set = []
    Identity = np.eye(8)
    for row in Identity:
        training_set.append({"mu": row})

    phi = fom.operator.source.empty(reserve=len(training_set))
    for mu in training_set:
        phi.append(fom.solve(mu))
    return phi


def restrict_to_omega(args, problem, snapshots):
    """restrict given snapshots to RVE domain omega

    Parameters
    ----------
    problem
        The RVE problem.
    snaphots
        The snapshots (block fom) which to restrict to omega.

    Returns
    -------
    rve_snapshots
        The snapshot data for the RVE.
    nodal_values
        Values at RVE vertices.
    """
    VV = snapshots.space.V
    V = problem.V
    S = FenicsVectorSpace(V)
    rve_nodes = np.array(
        [
            [problem.domain.xmin, problem.domain.ymin],
            [problem.domain.xmax, problem.domain.ymin],
            [problem.domain.xmax, problem.domain.ymax],
            [problem.domain.xmin, problem.domain.ymax],
        ]
    )

    f = df.Function(VV)
    rve_snapshots_data = []
    nodal_values = np.array([], dtype=np.float)
    for snapshot in snapshots._list:
        f.vector().set_local(snapshot.to_numpy())
        If = df.interpolate(f, V)
        for node in rve_nodes:
            nodal_values = np.append(nodal_values, If(node))
        rve_snapshots_data.append(If.vector().get_local())
    rve_snapshots = S.from_numpy(np.array(rve_snapshots_data))

    nodal_values.shape = (len(rve_snapshots), 8)
    return rve_snapshots, nodal_values


def test_PDE_locally_fulfilled(args, fom, problem, phi):
    """test that solutin of problem for omega with boundary
    data given as solution of FOM restricted to omega
    equals FOM solution restricted to omega"""
    rve_nodes = np.array(
        [
            [problem.domain.xmin, problem.domain.ymin],
            [problem.domain.xmax, problem.domain.ymin],
            [problem.domain.xmax, problem.domain.ymax],
            [problem.domain.xmin, problem.domain.ymax],
        ]
    )
    ps = fom.parameters.space(-1, 1)
    testing_set = ps.sample_randomly(1, seed=1)
    mu = testing_set[0]
    U = fom.solve(mu)
    W = U.space.V  # space for oversampling domain
    V = problem.V  # space for inner RVE
    u = df.Function(W)
    u.vector().set_local(U.to_numpy().flatten())
    s = df.interpolate(u, V)
    nodal_values = np.array([], dtype=np.float)
    for node in rve_nodes:
        nodal_values = np.append(nodal_values, s(node))
    nodal_values.shape = (1, 8)
    s_bilinear = phi.lincomb(nodal_values)
    g_bilinear = df.Function(V)  # dummy function to set bcs
    g_bilinear.vector().set_local(s_bilinear.to_numpy().flatten())

    def boundary(x, on_boundary):
        return on_boundary

    problem.bc_handler.add_bc(boundary, s)
    solver_parameters = args["solver"]["solver_parameters"]
    t = problem.solve(u=None, solver_parameters=solver_parameters)
    e = df.errornorm(s, t)
    if not e < 1e-9:
        print("test for global solution fullfilling PDE locally failed.")
        breakpoint()

    problem.bc_handler.remove_bcs()
    problem.bc_handler.add_bc(boundary, g_bilinear)
    t = problem.solve(u=None, solver_parameters=solver_parameters)
    e = df.errornorm(g_bilinear, t)
    if not e < 1e-9:
        print("test for bilinear part failed")
        breakpoint()


def plot_mode(basis, i, V):
    """plot x and y component of a 2d function
    as surfaces in 3d

    Parameters
    ----------
    basis
        The VectorArray.
    i
        Index of basis function.
    V
        dolfin FE space.
    """
    x_dofs = V.sub(0).collapse().tabulate_dof_coordinates()
    x = x_dofs[:, 0]
    y = x_dofs[:, 1]
    z = basis.to_numpy()[i]
    zx = z[::2]
    zy = z[1::2]
    plot_surface(1, x, y, zx)
    plot_surface(2, x, y, zy)
    plt.show()


def compute_psi(args, problem, boundary_data, V_to_L, product=None, method="trivial"):
    """compute psi with boundary data prescribed on edge

    Parameters
    ----------
    problem
        The linear variational problem.
    boundary_data : np.ndarray
        The boundary data to be prescribed on edge.
    V_to_L : int
        The map from V (rve space) to L (edge space).
    product
        The inner product wrt which psi is orthonormalized.
    method : str
        How to extend the VectorArray psi.

    Returns
    -------
    psi
        The VectorArray containing basis functions.

    """

    def boundary(x, on_boundary):
        return on_boundary

    V = problem.V
    u = df.Function(V)  # solution

    S = FenicsVectorSpace(V)
    psi = S.empty()

    g = df.Function(V)  # dirichlet data
    for m in range(len(boundary_data)):
        # reset bcs
        problem.bc_handler.remove_bcs()
        gvalues = np.zeros(V.dim())
        assert len(problem.bc_handler.bcs()) < 1

        gvalues[V_to_L] = boundary_data[m]
        g.vector().set_local(gvalues)

        problem.bc_handler.add_bc(boundary, g)
        solver_parameters = args["solver"]["solver_parameters"]
        # TODO how to check for df.parameters?
        breakpoint()
        problem.solve(u, solver_parameters=solver_parameters)

        d = u.copy(deepcopy=True)
        U = S.make_array([d.vector()])
        extend_basis(U, psi, product, method=method)

    return psi


def test_zero_bubble(args, problem, U_F):
    """compute s_b for a snapshot with exact psi"""
    domain = problem.domain
    V = problem.V

    S = FenicsVectorSpace(V)
    for j in range(len(U_F)):
        psi = S.empty()
        u_f = U_F[j]
        for i, edge in enumerate(domain.edges):
            L = df.VectorFunctionSpace(edge, "CG", args["DEG"], dim=2)
            V_to_L = make_mapping(L, V)
            edge_function = u_f.to_numpy()[:, V_to_L]
            psi_i = compute_psi(
                args, problem, edge_function, V_to_L, product=None, method="trivial"
            )
            psi.append(psi_i)
        s_b = u_f - psi.lincomb(np.ones((1, 4)))
        assert np.sum(s_b.amax()[1]) < 1e-8


def compute_edge_basis_via_pod(args, problem, U):
    """compute edge basis for bottom-top and right-left sets

    Parameters
    ----------
    problem
        The rve problem.
    U
        The fine scale snapshot data.
    """
    V = problem.V
    domain = problem.domain
    assert hasattr(domain, "edges")
    edges = domain.edges

    data = {0: [], 1: [], 2: [], 3: []}
    V_to_L = {}
    Lambda = {}

    for i, edge in enumerate(edges):
        for j, u in enumerate(U._list):
            L = df.VectorFunctionSpace(edge, "CG", args["DEG"], dim=2)
            Lambda[i] = L
            V_to_L[i] = make_mapping(L, V)
            edge_snapshot = u.to_numpy()[V_to_L[i]]
            data[i].append(edge_snapshot.copy())

    S = NumpyVectorSpace(L.dim())
    edge_basis = {}
    edge_basis[0] = S.empty()
    edge_basis[1] = S.empty()
    edge_basis[2] = S.empty()
    edge_basis[3] = S.empty()
    C = []

    bottom = S.make_array(data[0])
    right = S.make_array(data[1])
    top = S.make_array(data[2])
    left = S.make_array(data[3])

    if args["--pmax"] > 1:
        # FIXME pmax > 1 does not improve projection error decay ...
        from multi.shapes import mapping, get_hierarchical_shape_1d

        h = []
        x_dofs = Lambda[0].sub(0).collapse().tabulate_dof_coordinates()
        xi = mapping(x_dofs[:, 0], np.amin(x_dofs[:, 0]), np.amax(x_dofs[:, 0]))
        for deg in range(1, args["--pmax"]):
            f = get_hierarchical_shape_1d(deg)
            h.append(f(xi))
        H = S.make_array(np.kron(h, np.eye(2)))

        h_b_proj, h_b_coeff = compute_proj(H, bottom, None)
        h_r_proj, h_r_coeff = compute_proj(H, right, None)
        h_t_proj, h_t_coeff = compute_proj(H, top, None)
        h_l_proj, h_l_coeff = compute_proj(H, left, None)

        bottom -= H.lincomb(h_b_coeff.T)
        right -= H.lincomb(h_r_coeff.T)
        top -= H.lincomb(h_t_coeff.T)
        left -= H.lincomb(h_l_coeff.T)

        C.append(h_b_coeff)
        C.append(h_r_coeff)
        C.append(h_t_coeff)
        C.append(h_l_coeff)

        for i in range(4):
            edge_basis[i].append(H)

    # NOTE
    # to ensure interface continuity the same set of edge functions is
    # required for bottom-top, right-left respectively.
    # Thus the same snapshot data and resulting POD basis is used for the
    # respective sets.
    bt = S.empty()
    rl = S.empty()
    bt.append(bottom)
    bt.append(top)
    rl.append(right)
    rl.append(left)

    pod_basis_bt, svals_bt = pod(bt, product=None, rtol=args["--rtol"])
    pod_basis_rl, svals_rl = pod(rl, product=None, rtol=args["--rtol"])

    if args["--chi-svals"]:
        svals = np.vstack((svals_bt, svals_rl))
        np.save(args["--chi-svals"], svals)

    b_proj, b_coeff = compute_proj(pod_basis_bt, bottom, None)
    r_proj, r_coeff = compute_proj(pod_basis_rl, right, None)
    t_proj, t_coeff = compute_proj(pod_basis_bt, top, None)
    l_proj, l_coeff = compute_proj(pod_basis_rl, left, None)

    edge_basis[0].append(pod_basis_bt)
    edge_basis[1].append(pod_basis_rl)
    edge_basis[2].append(pod_basis_bt)
    edge_basis[3].append(pod_basis_rl)
    # FIXME if pmax > 1, edge_basis[i] is not orthonormal
    # Consider using GS to orthonormalize edge_basis[i] although it
    # does not change the span.
    #  for i in range(4):
    #      try:
    #          check_orthonormality(edge_basis[i], product=None)
    #      except AccuracyError:
    #          print("using gram schmidt to orthonormalize")
    #          gram_schmidt(edge_basis[i], product=None, copy=False)
    C.append(b_coeff)
    C.append(r_coeff)
    C.append(t_coeff)
    C.append(l_coeff)

    return edge_basis, V_to_L, np.vstack(C)


def check_interface_compatibility(psi, psi_other, mapping, mapping_other, ctol):
    modes_per_edge = min([len(psi), len(psi_other)])
    compatible = {"x": [], "y": []}
    for mode in range(modes_per_edge):
        edge = psi.to_numpy()[mode, mapping]
        other = psi_other.to_numpy()[mode, mapping_other]
        compatible["x"].append(np.allclose(edge[::2], other[::2], atol=ctol))
        compatible["y"].append(np.allclose(edge[1::2], other[1::2], atol=ctol))
    assert all(compatible["x"])
    assert all(compatible["y"])


def plot_surface(fid, x, y, z):
    plt.figure(fid)
    ax = plt.axes(projection="3d")
    ax.plot_trisurf(x, y, z, cmap="viridis", edgecolor="none")


def compute_proj(basis, V, product):
    G = basis.gramian(product=product)
    R = basis.inner(V, product=product)
    v = np.linalg.solve(G, R)
    return basis.lincomb(v.T), v


def compute_proj_errors(basis, V, product):
    G = basis.gramian(product=product)
    R = basis.inner(V, product=product)
    errors = []
    for N in range(len(basis) + 1):
        if N > 0:
            v = np.linalg.solve(G[:N, :N], R[:N, :])
        else:
            v = np.zeros((0, len(V)))
        V_proj = basis[:N].lincomb(v.T)
        #  try:
        #      n = (V - V_proj).norm(product=product)
        #  except RuntimeWarning as rw:
        #      breakpoint()
        relative = (V - V_proj).norm(product=product) / V.norm(product=product)
        errors.append(np.max(relative))
        #  errors.append(np.max((V - V_proj).norm(product=product)))
    return errors


def check_orthonormality(basis, product=None, offset=0, check_tol=1e-3):
    U = basis
    error_matrix = U[offset:].inner(U, product)
    error_matrix[: len(U) - offset, offset:] -= np.eye(len(U) - offset)
    if error_matrix.size > 0:
        err = np.max(np.abs(error_matrix))
        if err >= check_tol:
            raise AccuracyError(f"result not orthogonal (max err={err})")


def discretize_block(args, unit_length):
    """discretize the 3x3 block and wrap as pyMOR model"""
    block_domain = Domain(args["BLOCK"], 0, subdomains=True)
    material = args["material"]
    E = material["Material parameters"]["E"]["value"]
    NU = material["Material parameters"]["NU"]["value"]

    V = df.VectorFunctionSpace(block_domain.mesh, "CG", args["DEG"])
    problem = LinearElasticityProblem(block_domain, V, E=E, NU=NU, plane_stress=True)
    a = problem.get_lhs()

    A = df.assemble(a)
    A_bc = A.copy()

    # boundary data g
    q = df.Function(V)
    g = df.Function(V)

    def boundary(x, on_boundary):
        return on_boundary

    vector_operators = []
    null = np.zeros(V.dim())
    S = FenicsVectorSpace(V)
    x1 = block_domain.xmin
    x2 = block_domain.xmax
    y1 = block_domain.ymin
    y2 = block_domain.ymax
    quad8 = NumpyQuad(
        np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
                [x2 / 2, 0],
                [x2, y2 / 2],
                [x2 / 2, y2],
                [0, y2 / 2],
            ]
        )
    )

    shapes = quad8.interpolate(V.sub(0).collapse().tabulate_dof_coordinates(), (2,))
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
    opts = args["solver"]["solver_parameters"]
    solver_options = {
        "inverse": {
            "solver": opts["linear_solver"],
            "preconditioner": opts["preconditioner"],
        }
    }
    lhs = FenicsMatrixOperator(A_bc, V, V, solver_options=solver_options)

    # ### products
    energy_mat = A.copy()
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
            "l2": FenicsMatrixOperator(l2_mat, V, V, name="l2"),
            "l2_0": FenicsMatrixOperator(l2_0_mat, V, V, name="l2_0"),
            "h1": FenicsMatrixOperator(h1_mat, V, V, name="h1"),
            "h1_0": FenicsMatrixOperator(h1_0_mat, V, V, name="h1_0"),
        },
        estimator=None,
        visualizer=FenicsVisualizer(S),
        name="BLOCK",
    )
    return fom, problem


def discretize_rve(args):
    """discretize the rve and wrap as pyMOR model"""
    rve_domain = Domain(
        args["RVE"],
        id_=0,
        subdomains=True,
        edges=True,
        # need to know RVE unit length before I ccould compute it
        translate=df.Point((args["A"], args["A"])),
    )
    material = args["material"]
    E = material["Material parameters"]["E"]["value"]
    NU = material["Material parameters"]["NU"]["value"]

    V = df.VectorFunctionSpace(rve_domain.mesh, "CG", args["DEG"])
    problem = LinearElasticityProblem(rve_domain, V, E=E, NU=NU, plane_stress=True)
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
    shapes = quadrilateral.interpolate(
        V.sub(0).collapse().tabulate_dof_coordinates(), 2
    )
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
    opts = args["solver"]["solver_parameters"]
    solver_options = {
        "inverse": {
            "solver": opts["linear_solver"],
            "preconditioner": opts["preconditioner"],
        }
    }
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
        estimator=None,
        visualizer=FenicsVisualizer(S),
        name="RVE",
    )
    return fom, problem


def compute_snapshots(args, logger, fom, training_set):
    tic = time()
    snapshots = fom.operator.source.empty(reserve=len(training_set))
    for mu in training_set:
        snapshots.append(fom.solve(mu))
    elapsed_time = time() - tic

    summary = f"""{fom.name} snapshot set:
    size:          {len(snapshots)}
    elapsed time:  {elapsed_time}\n"""
    return snapshots, summary


def main(args):
    args = parse_arguments(args)
    logger = getLogger("empirical basis")
    logger.setLevel(args["--log"])

    # TODO better: one rve_problem which can be used to compute phi_i and psi_j?
    # only disadvantage is that computing psi does not fit in current pyMOR model
    rve_fom, rve_problem = discretize_rve(args)
    V = rve_problem.V
    with logger.block("Computing coarse scale basis phi ..."):
        phi = compute_coarse_scale_basis(args, rve_fom)

    block_fom, block_problem = discretize_block(args)
    if args["--test"]:
        test_PDE_locally_fulfilled(args, block_fom, rve_problem, phi)

    if args["--training-set"] == "random":
        logger.info("Using random sampling to generate training set ...")
        ps = block_fom.parameters.space(-1, 1)
        training_set = ps.sample_randomly(30, seed=4)
    elif args["--training-set"] == "delta":
        logger.info("Using unit vectors as training set ...")
        Identity = np.eye(16)
        training_set = []
        for row in Identity:
            training_set.append({"mu": row})
    else:
        # FIXME block uses hierarchical shape functions
        raise NotImplementedError("reimplement standard shape functions for block fom")

        logger.info("Reading training-set from file {args['--trainig-set']} ... ")
        training_set = []
        try:
            coefficients = np.load(args["--training-set"])
        except FileNotFoundError as exp:
            logger.warning(
                f"The training set {args['--training-set']} could not be found."
            )
            raise (exp)

        dofs_per_cell = 16
        ncells = int(coefficients.size / dofs_per_cell)
        for i in range(ncells):
            start = dofs_per_cell * i
            end = dofs_per_cell * (i + 1)
            mu = {"mu": coefficients[start:end]}
            training_set.append(mu)

    with logger.block("Solving on training set ..."):
        block_snapshots, block_summary = compute_snapshots(
            args, logger, block_fom, training_set
        )
        rve_snapshots, coarse_dofs = restrict_to_omega(
            args, rve_problem, block_snapshots
        )
        s_c = phi.lincomb(coarse_dofs)  # coarse scale part of snapshots
        s_f = rve_snapshots - s_c  # fine scale part of snapshots

    # this should be non-zero
    if args["--test"]:
        if not np.sum(s_f.amax()[1]) > 1e-2:
            plot_mode(s_f, 1, V)
            breakpoint()
        with logger.block("Running tests on zero bubble functions ..."):
            test_zero_bubble(args, rve_problem, s_f)

    chi, V_to_L, coefficients = compute_edge_basis_via_pod(args, rve_problem, s_f)

    if args["--chi"]:
        edge_basis = chi[0].copy()
        edge_basis.append(chi[1])
        np.save(args["--chi"], edge_basis.to_numpy())

    with logger.block("Computing psi_j ..."):
        psi_bottom = compute_psi(
            args,
            rve_problem,
            chi[0].to_numpy(),
            V_to_L[0],
            product=None,
            method="trivial",
        )
        psi_right = compute_psi(
            args,
            rve_problem,
            chi[1].to_numpy(),
            V_to_L[1],
            product=None,
            method="trivial",
        )
        psi_top = compute_psi(
            args,
            rve_problem,
            chi[2].to_numpy(),
            V_to_L[2],
            product=None,
            method="trivial",
        )
        psi_left = compute_psi(
            args,
            rve_problem,
            chi[3].to_numpy(),
            V_to_L[3],
            product=None,
            method="trivial",
        )
    if args["--check-interface"]:
        ctol = args["--check-interface"]
        with logger.block(f"Checking interface compatibility with tol={ctol} ... "):
            check_interface_compatibility(
                psi_bottom, psi_top, V_to_L[0], V_to_L[2], ctol
            )
            logger.info("Set bottom-top is okay.")
            check_interface_compatibility(
                psi_right, psi_left, V_to_L[1], V_to_L[3], ctol
            )
            logger.info("Set right-left is okay.")

    S = FenicsVectorSpace(rve_problem.V)
    basis = S.empty()
    basis.append(phi)
    max_psi = min([len(psi_bottom), len(psi_right), len(psi_top), len(psi_left)])
    for i in range(max_psi):
        basis.append(psi_bottom[i])
        basis.append(psi_right[i])
        basis.append(psi_top[i])
        basis.append(psi_left[i])

    if args["--output"]:
        np.save(args["--output"], basis.to_numpy())

    ps = block_fom.parameters.space(-1, 1)
    testing_set = ps.sample_randomly(10, seed=13)
    with logger.block("Solving on testing set ..."):
        test_snapshots, test_summary = compute_snapshots(
            args, logger, block_fom, testing_set
        )
        rve_test_snapshots, test_coarse_dofs = restrict_to_omega(
            args, rve_problem, test_snapshots
        )
    with logger.block("Computing projection errors ..."):
        errors = []
        names = []
        products = [None, rve_fom.energy_product, rve_fom.h1_product]
        for prod in products:
            # FIXME RuntimeError np.sqrt(norm_squared.real) for energy product
            # why should norm_squared.real be < 0?
            proj_errs = compute_proj_errors(basis, rve_test_snapshots, prod)
            try:
                name = prod.name
            except AttributeError:
                name = "euclidean"
            names.append(name)
            errors.append(proj_errs)

    if args["--plot-errors"]:
        plt.figure(12)
        plt.title("projection error")
        for err, name in zip(errors, names):
            nmodes = np.arange(len(err))
            plt.semilogy(nmodes + 1, err, "--*", label=f"{name}")
        reference = np.exp(-nmodes / 5)
        plt.semilogy(nmodes + 1, reference, "r--", label=r"$\exp(-n / 5)$")
        plt.grid()
        plt.legend()
        plt.show()

    if args["--projerr"]:
        with open(args["--projerr"], "w") as out:
            np.savetxt(out, np.vstack(errors).T, delimiter=",", header=", ".join(names))


if __name__ == "__main__":
    main(sys.argv[1:])
