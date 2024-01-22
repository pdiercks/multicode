import tempfile
from typing import Optional, Union, Callable
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
    # create_rectangle(0., 1., 0., 1., num_cells=4, recombine=True, out_file=fine.name)
    problem = MyProblem(coarse.name, fine.name)
    problem.setup_coarse_grid(2)
    # problem.setup_fine_grid() # FIXME: would need XDMF

    coarse.close()
    fine.close()

    problem.setup_coarse_space()
    # problem.setup_fine_space()

    # TODO: problem.build_edge_basis_config
    # TODO: problem.get_active_edges
