import dolfin as df
import numpy as np
from multi.bcs import compute_multiscale_bcs
from multi.dofmap import DofMap
from multi.domain import RectangularDomain
from multi.problems import LinearElasticityProblem
from multi.shapes import get_hierarchical_shape_functions
from helpers import build_mesh


def test():
    domain = RectangularDomain(
        "data/rvedomain.xdmf", _id=1, subdomains=False, edges=True
    )
    V = df.VectorFunctionSpace(domain.mesh, "CG", 1)
    problem = LinearElasticityProblem(domain, V, 210e3, 0.3)

    _ = build_mesh(1, 1, order=2, mshfile="data/test_multiscale_bcs.msh")
    dofmap = DofMap("data/test_multiscale_bcs.msh", tdim=2, gdim=2)

    """coarse grid / DofMap

    3------2       6,7-----4,5
    |      |        |      |
    |      |        |      |
    0------1       0,1-----2,3
                        8,9,10,11 for bottom edge

    physical space x in [0, 1]
    reference space ξ in [-1, 1]
    coarse: φ(ξ)=(ξ-1)/2 --> φ(x)=x
    fine: ψ(ξ)=ξ**2-1 --> ψ(x)=4(x**2-x)

    """

    # need to collapse function space
    x_dofs = problem.edge_spaces[0].sub(0).collapse().tabulate_dof_coordinates()
    pmax = 3
    dofmap.distribute_dofs(2, 2 * (pmax - 1), 0)
    hierarchical = get_hierarchical_shape_functions(x_dofs[:, 0], pmax, ncomp=2)
    edge_id = 0
    boundary_data = df.Expression(("x[0]", "0.0"), degree=1)
    bcs = compute_multiscale_bcs(
        problem, edge_id, boundary_data, dofmap, hierarchical, product=None, orth=False
    )
    assert np.allclose(
        np.array(list(bcs.keys())),
        dofmap.locate_dofs([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]),
    )
    values = np.zeros(8)
    values[2] = 1.0
    assert np.allclose(np.array(list(bcs.values())), values)

    boundary_data = df.Expression(("0.0", "x[0] * x[0]"), degree=2)
    bcs = compute_multiscale_bcs(
        problem, edge_id, boundary_data, dofmap, hierarchical, product=None, orth=False
    )
    assert np.allclose(
        np.array(list(bcs.keys())),
        dofmap.locate_dofs([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]),
    )
    values = np.zeros(8)
    values[3] = 1.0  # y-component for linear shape function φ(x)=x
    values[5] = (
        1.0 / 4.0
    )  # corresponds to dof 9; y-component for hierarchical shape function ψ(x)=4(x**2-x)
    assert np.allclose(np.array(list(bcs.values())), values)


if __name__ == "__main__":
    test()
