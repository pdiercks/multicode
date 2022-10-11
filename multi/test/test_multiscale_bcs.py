import tempfile
import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from multi.bcs import compute_multiscale_bcs
from multi.dofmap import DofMap
from multi.domain import RectangularDomain
from multi.problems import LinearElasticityProblem
from multi.preprocessing import (
    create_rce_grid_01,
    create_line_grid,
    create_rectangle_grid,
)
from multi.shapes import get_hierarchical_shape_functions


def test():
    # create rce grid
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rce_grid_01(0.0, 1.0, 0.0, 1.0, num_cells_per_edge=10, out_file=tf.name)
        rce_mesh, cell_markers, facet_markers = gmshio.read_from_msh(
            tf.name, MPI.COMM_WORLD, gdim=2
        )

    # create edge grids
    # my_edges = {}
    # edges = ["bottom", "right", "top", "left"]
    # points = [
    #         ([0, 0, 0], [1, 0, 0]),
    #         ([1, 0, 0], [1, 1, 0]),
    #         ([0, 1, 0], [1, 1, 0]),
    #         ([0, 0, 0], [0, 1, 0]),
    #         ]
    # for name, (start, end) in zip(edges, points):
    #     with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
    #         create_line_grid(start, end, num_cells=10, out_file=tf.name)
    #         line, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)
    #         my_edges[name] = line

    # FIXME there may be an issue with line grid creation
    # try submesh instead ...

    def bottom(x):
        return np.isclose(x[1], 0.0)

    bottom_entities = dolfinx.mesh.locate_entities_boundary(rce_mesh, 1, bottom)
    bottom_mesh = dolfinx.mesh.create_submesh(rce_mesh, 1, bottom_entities)[0]
    # FIXME this does not work either ...
    vertices = dolfinx.mesh.exterior_facet_indices(bottom_mesh.topology)

    breakpoint()
    rce_domain = RectangularDomain(
        rce_mesh, cell_markers, facet_markers, index=1, edges=my_edges
    )
    V = dolfinx.fem.VectorFunctionSpace(rce_domain.mesh, ("CG", 1))
    problem = LinearElasticityProblem(
        rce_domain, V, [30e3, 60e3], [0.2, 0.2], plane_stress=True
    )
    # TODO edge spaces are actually vector elements of dim=2

    # TODO create coarse grid
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle_grid(
            0.0, 1.0, 0.0, 1.0, num_cells=1, recombine=True, out_file=tf.name
        )
        coarse_grid, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    # grid = StructuredQuadGrid(coarse_grid)
    dofmap = DofMap(coarse_grid)

    ndofs_ent = (2, 4, 0)
    dofmap.distribute_dofs(*ndofs_ent)
    pmax = int(ndofs_ent[1] / 2 + 1)

    """DofMap

    1------3       2,3-----6,7
    |      |        |      |
    |      |        |      |
    0------2       0,1-----4,5

            8,  9, 10, 11 for left edge
           12, 13, 14, 15 for bottom edge
           16, 17, 18, 19 for top edge
           20, 21, 22, 23 for right edge


    physical space x in [0, 1]
    reference space ξ in [-1, 1]
    coarse: φ(ξ)=(ξ-1)/2 --> φ(x)=x
    fine: ψ(ξ)=ξ**2-1 --> ψ(x)=4(x**2-x)

    """

    # need to collapse function space
    x_dofs = problem.edge_spaces["bottom"].tabulate_dof_coordinates()
    hierarchical = get_hierarchical_shape_functions(x_dofs[:, 0], pmax, ncomp=2)
    cell_index = 0
    edge_id = 0
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
    assert np.allclose(np.array(list(bcs.keys())), np.array([12, 13, 14, 15]))
    values = np.zeros(8)
    values[2] = 1.0
    assert np.allclose(np.array(list(bcs.values())), values)

    boundary_data = dolfinx.fem.Function(V)
    boundary_data.interpolate(lambda x: (np.zeros_like(x[0]), x[0] * x[0]))
    # left, bottom, top, right
    dofs_per_edge = np.array([[3, 2 * (pmax - 1), 2, 5]])
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
    assert np.allclose(
        np.array(list(bcs.keys())),
        # left: 8, 9, 10
        np.array([11, 12, 13, 14]),
    )
    values = np.zeros(8)
    values[3] = 1.0  # y-component for linear shape function φ(x)=x
    values[5] = (
        1.0 / 4.0
    )  # corresponds to dof 9; y-component for hierarchical shape function ψ(x)=4(x**2-x)
    assert np.allclose(np.array(list(bcs.values())), values)


if __name__ == "__main__":
    test()
