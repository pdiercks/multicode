"""test Quadrilateral"""

import pytest
from numpy import arange, array, allclose
from multi.dofmap import Quadrilateral


@pytest.mark.parametrize(
    "ctype,d_to_ndofs,d_to_nent,gmsh_cell,d_to_ent",
    [
        (
            "quad",
            {0: 2, 1: 0, 2: 0},
            {0: 4, 1: 0, 2: 0},
            arange(4),
            {0: arange(4), 1: array([]), 2: array([])},
        ),
        (
            "quad8",
            {0: 2, 1: 5, 2: 0},
            {0: 4, 1: 4, 2: 0},
            arange(8),
            {0: arange(4), 1: arange(4, 8), 2: array([])},
        ),
        (
            "quad9",
            {0: 2, 1: 2, 2: 3},
            {0: 4, 1: 4, 2: 1},
            arange(9),
            {0: arange(4), 1: arange(4, 8), 2: array([8])},
        ),
    ],
)
def test(ctype, d_to_ndofs, d_to_nent, gmsh_cell, d_to_ent):
    cell = Quadrilateral(ctype)
    assert len(cell.verts) == 4

    with pytest.raises(AttributeError):
        cell.get_entity_dofs()
        cell.get_entities()

    cell.set_entity_dofs(d_to_ndofs[0], d_to_ndofs[1], d_to_ndofs[2])
    edofs = cell.get_entity_dofs()
    cell.set_entities(gmsh_cell)
    entities = cell.get_entities()

    for dim in [0, 1, 2]:
        assert len(edofs[dim].keys()) == d_to_nent[dim]
        if len(edofs[dim].keys()) > 0:
            assert len(edofs[dim][0]) == d_to_ndofs[dim]

        allclose(entities[dim], d_to_ent[dim])


if __name__ == "__main__":
    test()
