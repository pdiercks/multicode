"""
decompose a VectorArray into coarse and fine scale parts

Usage:
    decompose_vectorarray.py [options] OMEGA DEGREE MATERIAL ARRAY

Arguments:
    OMEGA        The computational domain.
    DEGREE       The order of the FE shape functions.
    MATERIAL     The material parameters.
    ARRAY        The VectorArray data (.npy).

Options:
    -h, --help               Show this message and exit.
    -o FILE, --output=FILE   Write decomposed basis to FILE.
"""

import sys
import yaml
from docopt import docopt
from pathlib import Path
import numpy as np
import dolfin as df
from multi import Domain, LinearElasticityProblem
from multi.basis_construction import construct_coarse_scale_basis
from multi.misc import make_mapping, locate_dofs, get_solver
from multi.extension import extend

default_solver = get_solver(None)
prm = df.parameters
prm["krylov_solver"]["relative_tolerance"] = default_solver["krylov_solver"][
    "relative_tolerance"
]
prm["krylov_solver"]["absolute_tolerance"] = default_solver["krylov_solver"][
    "absolute_tolerance"
]
prm["krylov_solver"]["maximum_iterations"] = default_solver["krylov_solver"][
    "maximum_iterations"
]


def parse_args(args):
    args = docopt(__doc__, args)
    args["OMEGA"] = Path(args["OMEGA"])
    args["DEGREE"] = int(args["DEGREE"])
    args["MATERIAL"] = Path(args["MATERIAL"])
    args["ARRAY"] = Path(args["ARRAY"])
    return args


def main(args):
    args = parse_args(args)

    domain = Domain(args["OMEGA"], 1, subdomains=True, edges=True)
    V = df.VectorFunctionSpace(domain.mesh, "CG", args["DEGREE"])
    with args["MATERIAL"].open() as handle:
        material = yaml.safe_load(handle)
    E = material["Material parameters"]["E"]["value"]
    NU = material["Material parameters"]["NU"]["value"]
    plane_stress = material["Constraints"]["plane_stress"]
    problem = LinearElasticityProblem(domain, V, E=E, NU=NU, plane_stress=plane_stress)
    # coarse scale basis
    phi = construct_coarse_scale_basis(problem)
    space = phi.space

    # VectorArray
    va = np.load(args["ARRAY"])
    U = space.from_numpy(va)

    # subtract coarse scale part
    vertices = domain.get_nodes(n=4)
    vertex_dofs = locate_dofs(V.tabulate_dof_coordinates(), vertices)
    nodal_values = U.dofs(vertex_dofs)
    U -= phi.lincomb(nodal_values)
    assert np.sum(U.dofs(vertex_dofs)) < 1e-6

    # restrict to edges
    Lambda = {}
    V_to_L = {}
    edge_functions = {}
    element = V.ufl_element()
    for i, edge in enumerate(domain.edges):
        L = df.VectorFunctionSpace(edge, element.family(), element.degree())
        Lambda[i] = L
        V_to_L[i] = make_mapping(L, problem.V)
        zeta = U.dofs(V_to_L[i])
        edge_functions[i] = zeta

    # extend edge functions into domain separately
    boundary_data = []
    for i in range(4):
        for mode in edge_functions[i]:
            g = df.Function(problem.V)
            gvec = g.vector()
            gvec.zero()
            gvec[V_to_L[i]] = mode
            boundary_data.append(g)
    psi = extend(
        problem,
        boundary_data,
        solver_options={"linear_solver": "default", "preconditioner": "default"},
    )
    PSI = space.make_array(psi)
    zeros = PSI.dofs(vertex_dofs)
    assert np.sum(zeros) < 1e-6

    # make test
    reference = space.from_numpy(va)
    v_values = reference.dofs(vertex_dofs)
    coarse_part = phi.lincomb(v_values)
    N = int(len(PSI) / 4)  # == 58
    R = coarse_part + PSI[:N] + PSI[N : 2 * N] + PSI[2 * N : 3 * N] + PSI[3 * N :]
    err = reference - R
    assert np.all(err.norm() < 1e-2)

    if args["--output"] is not None:
        N = int(len(PSI) / 4)
        np.savez(
            args["--output"],
            phi=phi.to_numpy(),
            b=PSI[:N].to_numpy(),
            r=PSI[N : 2 * N].to_numpy(),
            t=PSI[2 * N : 3 * N].to_numpy(),
            l=PSI[3 * N :].to_numpy(),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
