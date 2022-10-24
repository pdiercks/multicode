import tempfile
import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from multi.bcs import compute_multiscale_bcs
from multi.dofmap import DofMap
from multi.domain import RceDomain, StructuredQuadGrid
from multi.problems import LinearElasticityProblem
from multi.preprocessing import (
    create_rce_grid_01,
    create_rectangle_grid,
)
from multi.shapes import get_hierarchical_shape_functions


def test():
    # create rce grid
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rce_grid_01(0.0, 1.0, 0.0, 1.0, num_cells=10, out_file=tf.name)
        rce_mesh, cell_markers, facet_markers = gmshio.read_from_msh(
            tf.name, MPI.COMM_WORLD, gdim=2
        )

    rce_domain = RceDomain(rce_mesh, cell_markers, facet_markers, index=1)
    rce_domain.create_edge_meshes(10)
    V = dolfinx.fem.VectorFunctionSpace(rce_domain.mesh, ("Lagrange", 1))
    problem = LinearElasticityProblem(
        rce_domain, V, [30e3, 60e3], [0.2, 0.2], plane_stress=True
    )
    bottom_space = problem.edge_spaces["bottom"]
    bottom_el = bottom_space.ufl_element()
    assert bottom_el.degree() == 1
    assert bottom_el.family() in ("Lagrange", "P")
    assert bottom_el.value_shape()[0] == 2

    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(
            0.0, 1.0, 0.0, 1.0, num_cells=1, recombine=True, out_file=tf.name
        )
        coarse_grid, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    cgrid = StructuredQuadGrid(coarse_grid)
    dofmap = DofMap(cgrid)

    ndofs_ent = (2, 4, 0)
    dofmap.distribute_dofs(*ndofs_ent)
    pmax = int(ndofs_ent[1] / 2 + 1)

    """DofMap

    2------3       4,5-----6,7
    |      |        |      |
    |      |        |      |
    0------1       0,1-----2,3

            see DofMap.QuadrilateralCellLayout
            for ordering of edges
            8,  9, 10, 11 for bottom edge
           12, 13, 14, 15 for left edge
           16, 17, 18, 19 for right edge
           20, 21, 22, 23 for top edge


    physical space x in [0, 1]
    reference space ξ in [-1, 1]
    coarse: φ(ξ)=(ξ-1)/2 --> φ(x)=x
    fine: ψ(ξ)=ξ**2-1 --> ψ(x)=4(x**2-x)

    """

    # need to collapse function space
    x_dofs = problem.edge_spaces["bottom"].tabulate_dof_coordinates()
    hierarchical = get_hierarchical_shape_functions(x_dofs[:, 0], pmax, ncomp=2)
    cell_index = 0
    edge_id = "bottom"
    boundary_data = dolfinx.fem.Function(V)
    boundary_data.interpolate(lambda x: (x[0], np.zeros_like(x[0])))
    bcs = compute_multiscale_bcs(
        problem,
        cell_index,
        edge_id,
        boundary_data,
        dofmap,
        hierarchical,
        product=None,
        orth=False,
    )
    # expect: 2 * 2 coarse dofs, and 1 * 4 fine dofs
    assert len(bcs.keys()) == 8
    assert np.allclose(np.array(list(bcs.keys())), np.array([0, 1, 2, 3, 8, 9, 10, 11]))
    values = np.zeros(8)
    values[2] = 1.0
    assert np.allclose(np.array(list(bcs.values())), values)

    boundary_data = dolfinx.fem.Function(V)
    boundary_data.interpolate(lambda x: (np.zeros_like(x[0]), x[0] * x[0]))
    # distribute dofs in order bottom, left, right, top
    dofs_per_edge = np.array([[2 * (pmax - 1), 3, 5, 2]])
    dofmap.distribute_dofs(2, dofs_per_edge, 0)
    bcs = compute_multiscale_bcs(
        problem,
        cell_index,
        edge_id,
        boundary_data,
        dofmap,
        hierarchical,
        product=None,
        orth=False,
    )
    # expected: again 4 coarse dofs, and 2 * (pmax-1) fine dofs
    assert np.allclose(
        np.array(list(bcs.keys())),
        np.array([0, 1, 2, 3, 8, 9, 10, 11]),
    )
    values = np.zeros(8)
    values[3] = 1.0  # y-component for linear shape function φ(x)=x
    values[5] = (
        1.0 / 4.0
    )  # corresponds to dof 9; y-component for hierarchical shape function ψ(x)=4(x**2-x)
    assert np.allclose(np.array(list(bcs.values())), values)


if __name__ == "__main__":
    test()
