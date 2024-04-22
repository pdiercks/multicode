from mpi4py import MPI
from dolfinx.io import gmshio
import tempfile
from multi.preprocessing import merge_mshfiles


def test_merge_unit_cell_01():
    from multi.preprocessing import create_unit_cell_01

    coord = [[0.0, 1.0, 0.0, 1.0], [1.0, 2.0, 0.0, 1.0]]
    num_cells = 2
    to_be_merged = []
    tfiles = []
    tfiles.append(tempfile.NamedTemporaryFile(suffix=".msh"))
    tfiles.append(tempfile.NamedTemporaryFile(suffix=".msh"))

    cell_tags = [{"matrix": 1, "inclusion": 3}, {"matrix": 2, "inclusion": 8}]
    offset = {2: 0}
    for i in range(num_cells):
        tmp = tfiles[i]
        xmin, xmax, ymin, ymax = coord[i]
        create_unit_cell_01(
            xmin,
            xmax,
            ymin,
            ymax,
            num_cells=10,
            recombine=False,
            cell_tags=cell_tags[i],
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

    mesh, ct, _ = gmshio.read_from_msh(final.name, MPI.COMM_WORLD, gdim=2)
    final.close()

    assert mesh.topology.dim == 2
    assert ct.find(0).size == 0
    assert ct.find(cell_tags[0]["matrix"]).size > 0
    assert ct.find(cell_tags[1]["matrix"]).size > 0
    assert ct.find(cell_tags[0]["inclusion"]).size > 0
    assert ct.find(cell_tags[1]["inclusion"]).size > 0
