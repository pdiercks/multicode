import tempfile
from typing import Optional, Union, Callable

from mpi4py import MPI
from dolfinx import default_scalar_type

import numpy as np

from multi.boundary import plane_at
from multi.preprocessing import create_rectangle
from multi.problems import MultiscaleProblemDefinition


# Define custom problem
# override methods that need to be overridden
class MyProblem(MultiscaleProblemDefinition):
    def __init__(self, cg, fg):
        super().__init__(cg, fg)

    @property
    def boundaries(self) -> dict[str, tuple[int, Callable]]:
        return {
                "bottom": (int(11), plane_at(0., "y")),
                "left": (int(12), plane_at(0., "x")),
                "right": (int(13), plane_at(1., "x")),
                "top": (int(14), plane_at(1., "y")),
                }

    def get_dirichlet(self, cell_index: Optional[int] = None) -> Union[dict, None]:
        _, bottom = self.boundaries["bottom"]
        return {"value": np.array([0, 0], dtype=default_scalar_type), "boundary": bottom, "method": "geometrical"}

    def get_neumann(self, cell_index: Optional[int] = None) -> Union[dict, None]:
        return None

    def get_kernel_set(self, cell_index: int) -> tuple[int, ...]:
        if cell_index == 0:
            return ()
        else:
            return (0, 1, 2)

    def get_gamma_out(self, cell_index: int) -> Callable:
        everywhere = lambda x: np.full(x.shape[1], True, dtype=bool)
        return everywhere

    @property
    def cell_sets(self) -> dict[str, set[int]]:
        cells = {"inner": set([0, 1, 2]), "right": set([3, 4, 5])}
        return cells


def test():
    coarse = tempfile.NamedTemporaryFile(suffix='.msh')
    fine = tempfile.NamedTemporaryFile(suffix='.msh')
    create_rectangle(0., 1., 0., 1., num_cells=2, recombine=True, out_file=coarse.name)
    """

    coarse grid cells
    |----|----|
    | 2  | 3  |
    |----|----|
    | 0  | 1  |
    |----|----|

    """

    problem = MyProblem(coarse.name, fine.name)
    problem.setup_coarse_grid(MPI.COMM_SELF, 2)
    # problem.setup_fine_grid() # FIXME: would need XDMF

    coarse.close()
    fine.close()

    problem.setup_coarse_space()
    # problem.setup_fine_space()

    cell_sets = {"A": set([0, 1]), "B": set([2, 3])}
    problem.build_edge_basis_config(cell_sets)
    # due to order, cells 0 and 1 should own its top edges 4 and 6 respectively
    # cell 0, also owns edge that is shared with cell 1
    edges_0 = problem.active_edges(0)
    edges_1 = problem.active_edges(1)
    edges_2 = problem.active_edges(2)
    edges_3 = problem.active_edges(3)

    assert len(edges_0) == 4
    assert len(edges_1) == 3
    assert len(edges_2) == 3
    assert len(edges_3) == 2

    assert "bottom" not in edges_2
    assert "bottom" not in edges_3
    assert "left" not in edges_3
    assert "left" not in edges_1

    (cell_0, top) = problem.edge_to_cell(4)
    assert cell_0 == 0
    assert top == "top"
    (cell_2, right) = problem.edge_to_cell(7)
    assert cell_2 == 2
    assert right == "right"


if __name__ == "__main__":
    test()
