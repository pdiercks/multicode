"""
basis construction

(1) phi_i: bilinear functions which also account for the inheterogenity
(2) psi_j: edge functions (=hierarchical shape functions) which also account for the inheterogenity

Usage:
    generate_hierarchical_basis.py [options] RVE A DEG MAT

Arguments:
    RVE                 XDMF file path (incl. extension).
    A                   The unit length of the RVE.
    DEG                 Degree of (fine grid) FE space.
    MAT                 The material parameters (.yml).

Options:
    -h, --help               Show this message.
    -l LEVEL, --log=LEVEL    Set the log level [default: 30].
    --solver=SOLVER          Define dolfin solver parameters.
    --pmax=PMAX              Use hierarchical polynomials up to pmax degree as set of edge functions.
    -o FILE, --output=FILE   Specify output path for basis (.npy).
    --chi=FILE               Write edge basis functions to path (.npz).
    --check-interface=TOL    Check interface compatibility with tolerance TOL.
"""

import sys
from pathlib import Path

import yaml
import dolfin as df
import numpy as np
from docopt import docopt

from multi import Domain, LinearElasticityProblem, make_mapping
from multi.shapes import NumpyQuad
from multi.misc import get_solver

from pymor.bindings.fenics import (
    FenicsMatrixOperator,
    FenicsVectorSpace,
    FenicsVisualizer,
)
from pymor.core.logger import getLogger
from pymor.models.basic import StationaryModel
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.basic import extend_basis


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["RVE"] = Path(args["RVE"])
    args["A"] = float(args["A"])
    args["DEG"] = int(args["DEG"])
    with open(Path(args["MAT"]), "r") as infile:
        try:
            args["material"] = yaml.safe_load(infile)
        except yaml.YAMLError as exc:
            print(exc)
    args["--pmax"] = int(args["--pmax"]) if args["--pmax"] else None
    args["--log"] = int(args["--log"])
    args["--output"] = Path(args["--output"]) if args["--output"] else None
    args["--chi"] = Path(args["--chi"]) if args["--chi"] else None
    args["--check-interface"] = (
        float(args["--check-interface"]) if args["--check-interface"] else None
    )
    args["solver"] = get_solver(args["--solver"])
    prm = df.parameters
    prm["krylov_solver"]["relative_tolerance"] = args["solver"]["krylov_solver"][
        "relative_tolerance"
    ]
    prm["krylov_solver"]["absolute_tolerance"] = args["solver"]["krylov_solver"][
        "absolute_tolerance"
    ]
    prm["krylov_solver"]["maximum_iterations"] = args["solver"]["krylov_solver"][
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
        problem.solve(u, solver_parameters=solver_parameters)

        d = u.copy(deepcopy=True)
        U = S.make_array([d.vector()])
        extend_basis(U, psi, product, method=method)

    return psi


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
    # TODO use multi.shapes.get_hierarchical_shapes_2d with optional argument PMAX
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


def compute_hierarchical_edge_basis(args, problem):
    """compute hierarchical edge basis for each edge

    Parameters
    ----------
    problem
        The RVE problem.

    """
    from multi.shapes import get_hierarchical_shape_1d, mapping

    V = problem.V
    domain = problem.domain
    assert hasattr(domain, "edges")
    edges = domain.edges

    ncomp = V.num_sub_spaces()
    L = df.FunctionSpace(edges[0], "CG", args["DEG"])
    S = NumpyVectorSpace(L.dim() * ncomp)
    x_dofs = L.tabulate_dof_coordinates()
    sub = 0
    xi = mapping(x_dofs[:, sub], np.amin(x_dofs[:, sub]), np.amax(x_dofs[:, sub]))
    h = []
    for deg in range(2, args["--pmax"] + 1):
        f = get_hierarchical_shape_1d(deg)
        h.append(f(xi))
    edge_basis = S.make_array(np.kron(h, np.eye(ncomp)))
    # gram_schmidt(edge_basis, product=None, copy=False, rtol=args["--rtol"])
    # len(edge_basis) <-- ncomp * (PMAX)
    return edge_basis


def main(args):
    args = parse_arguments(args)
    logger = getLogger("basis construction")
    logger.setLevel(args["--log"])

    rve_fom, rve_problem = discretize_rve(args)
    V = rve_problem.V
    with logger.block("Computing coarse scale basis phi ..."):
        phi = compute_coarse_scale_basis(args, rve_fom)

    with logger.block("Computing edge space mappings ..."):
        V_to_L = {}
        Lambda = {}
        for i, edge in enumerate(rve_problem.domain.edges):
            L = df.VectorFunctionSpace(edge, "CG", args["DEG"], dim=2)
            Lambda[i] = L
            V_to_L[i] = make_mapping(L, V)

    # set hierarchical basis
    with logger.block("Computing hierarchical edge functions ..."):
        hier_chi = compute_hierarchical_edge_basis(args, rve_problem)
        chi = {
            0: hier_chi,
            1: hier_chi,
            2: hier_chi,
            3: hier_chi,
        }

    if args["--chi"]:
        # save arrays separately
        np.savez(args["--chi"], **{"0": chi[0].to_numpy(), "1": chi[1].to_numpy()})

    with logger.block("Computing psi_j ..."):
        psi = []
        for i in range(len(rve_problem.domain.edges)):
            psi_ = compute_psi(
                args,
                rve_problem,
                chi[i].to_numpy(),
                V_to_L[i],
                product=None,
                method="trivial",
            )
            psi.append(psi_)

    if args["--check-interface"]:
        ctol = args["--check-interface"]
        with logger.block(f"Checking interface compatibility with tol={ctol} ... "):
            check_interface_compatibility(psi[0], psi[2], V_to_L[0], V_to_L[2], ctol)
            logger.info("Set bottom-top is okay.")
            check_interface_compatibility(psi[1], psi[3], V_to_L[1], V_to_L[3], ctol)
            logger.info("Set right-left is okay.")

    if args["--output"]:
        np.savez(
            args["--output"],
            phi=phi.to_numpy(),
            b=psi[0].to_numpy(),
            r=psi[1].to_numpy(),
            t=psi[2].to_numpy(),
            l=psi[3].to_numpy(),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
