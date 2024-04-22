from mpi4py import MPI
from dolfinx.io import gmshio
import tempfile
import pytest
from multi.preprocessing import merge_mshfiles


@pytest.mark.parametrize("method", ["uc01", "void"])
def test_merge(method):
    if method == "uc01":
        from multi.preprocessing import create_unit_cell_01 as create_grid
        cell_tags = [{"matrix": 1, "inclusion": 3}, {"matrix": 2, "inclusion": 8}]
        facet_tags = [{"bottom": 11, "left": 12, "right": 13, "top": 14}, {"bottom": 11, "left": 122, "right": 133, "top": 14}]
    elif method == "void":
        from multi.preprocessing import create_voided_rectangle as create_grid
        cell_tags = [{"matrix": 1}, {"matrix": 2}]
        facet_tags = [{"bottom": 11, "left": 12, "right": 13, "top": 14, "void": 98}, {"bottom": 11, "left": 122, "right": 133, "top": 14, "void": 1002}]
    else:
        raise NotImplementedError

    coord = [[0.0, 1.0, 0.0, 1.0], [1.0, 2.0, 0.0, 1.0]]
    num_coarse_cells = 2
    num_cells = 10
    to_be_merged = []
    tfiles = []
    tfiles.append(tempfile.NamedTemporaryFile(suffix=".msh"))
    tfiles.append(tempfile.NamedTemporaryFile(suffix=".msh"))

    offset = {2: 0, 1: 0}
    for i in range(num_coarse_cells):
        tmp = tfiles[i]
        xmin, xmax, ymin, ymax = coord[i]
        create_grid(
            xmin,
            xmax,
            ymin,
            ymax,
            num_cells=num_cells,
            recombine=False,
            cell_tags=cell_tags[i],
            facet_tags=facet_tags[i],
            out_file=tmp.name,
            tag_counter=offset
        )
        to_be_merged.append(tmp.name)
    final = tempfile.NamedTemporaryFile(suffix=".msh")
    merge_mshfiles(to_be_merged, final.name)

    # by providing tag_counter=offset we ensure that the `tag` is
    # different for each surface etc. in the two meshes that are created.
    # If the tags are not unique `gmshio` will not be able to input
    # the merged mesh correctly.

    for tf in tfiles:
        tf.close()

    mesh, ct, ft = gmshio.read_from_msh(final.name, MPI.COMM_WORLD, gdim=2)
    final.close()

    assert mesh.topology.dim == 2
    assert ct.find(0).size == 0
    for cell in range(num_coarse_cells):
        for tag in cell_tags[cell].values():
            assert ct.find(tag).size > 0

    assert ft.find(11).size == 2 * num_cells # bottom
    assert ft.find(14).size == 2 * num_cells # top
    assert ft.find(12).size == num_cells # left
    assert ft.find(133).size == num_cells # right
    assert ft.find(122).size == 0 # removed during merge, because duplicate
    assert ft.find(13).size == num_cells
    if method == "void":
        assert ft.find(98).size > 0
        assert ft.find(1002).size > 0
