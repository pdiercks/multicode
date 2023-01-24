import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
import yaml

from multi.problems import LinearElasticityProblem
from multi.domain import RectangularDomain
from multi.bcs import BoundaryDataFactory
from multi.extension import extend

# from multi.misc import locate_dofs
# from multi.product import InnerProduct
from multi.shapes import NumpyQuad  # , get_hierarchical_shape_functions

# from scipy.sparse.linalg import eigsh, LinearOperator
# from scipy.special import erfinv

# from pymor.algorithms.gram_schmidt import gram_schmidt
# from pymor.bindings.fenics import FenicsVectorSpace
# from pymor.operators.interface import Operator
# from pymor.vectorarrays.interface import VectorArray


def compute_phi(problem, nodes):
    """compute coarse scale basis functions for given problem"""
    V = problem.V
    quadrilateral = NumpyQuad(nodes)
    shape_functions = quadrilateral.interpolate(V)

    data_factory = BoundaryDataFactory(problem.domain.grid, V)

    boundary_data = []
    g = dolfinx.fem.Function(V)
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


def compute_coarse_scale_basis(rce_grid, material, degree, out_file):
    """compute the coarse scale basis (extension of bilinear shape functions)

    NOTE
    ----
    method to be used within python action of a dodoFile

    Parameters
    ----------
    rce_grid : filepath
        The partition of the subdomain.
    material : filepath
        The material parameters (.yaml).
    degree : int
        Degree of the VectorFunctionSpace
    """
    domain, cell_marker, facet_marker = gmshio.read_from_msh(
        rce_grid, MPI.COMM_SELF, gdim=2
    )
    omega = RectangularDomain(domain, cell_marker, facet_marker, index=0)
    V = dolfinx.fem.VectorFunctionSpace(domain, ("Lagrange", degree))

    # FIXME define nodes via multi.dofmap.QuadrilateralDofLayout ?
    xmin = omega.xmin
    xmax = omega.xmax
    nodes = np.array([
        [xmin[0], xmin[1], 0.],
        [xmax[0], xmin[1], 0.],
        [xmin[0], xmax[1], 0.],
        [xmax[0], xmax[1], 0.]
        ])

    with material.open("r") as f:
        mat = yaml.safe_load(f)

    E = mat["Material parameters"]["E"]["value"]
    NU = mat["Material parameters"]["NU"]["value"]
    plane_stress = mat["Constraints"]["plane_stress"]
    problem = LinearElasticityProblem(omega, V, E=E, NU=NU, plane_stress=plane_stress)
    basis_vectors = compute_phi(problem, nodes)
    out = []
    for vec in basis_vectors:
        out.append(vec.array)

    np.savez(out_file, phi=out)
