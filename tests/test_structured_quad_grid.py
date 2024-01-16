from mpi4py import MPI
import pathlib
import tempfile
import pytest
import numpy as np
from dolfinx.io import gmshio
from dolfinx.io.utils import XDMFFile
from multi.domain import StructuredQuadGrid
from multi.preprocessing import create_rectangle, create_unit_cell_01, create_line


def test():
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(
            0.0, 1.0, 0.0, 1.0, num_cells=(10, 10), recombine=True, out_file=tf.name
        )
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    grid = StructuredQuadGrid(domain)
    patch_sizes = []
    for i in range(100):
        cell_patch = grid.get_patch(i)
        patch_sizes.append(cell_patch.size)
    assert np.amin(patch_sizes) == 4
    assert np.amax(patch_sizes) == 9
    num_inner = 100 - 4 - 4 * 8
    expected = num_inner * 9 + 4 * 4 + 4 * 8 * 6
    assert np.sum(patch_sizes) == expected

    num_entities = (4, 4)
    for dim, num in enumerate(num_entities):
        ents = grid.get_entities(dim, 8)
        assert ents.size == num

    verts = grid.get_entities(0, 67)  # assume cell 67 is surrounded by 8 cells
    cells_67 = grid.get_cells(0, verts)
    assert cells_67.size == 9
    assert np.allclose(cells_67, grid.get_patch(67))

    cell_index = 99
    verts = grid.get_entities(0, cell_index)
    x_verts = grid.get_entity_coordinates(0, verts)
    assert np.amin(x_verts[:, 0]) > 0.8
    assert np.amin(x_verts[:, 1]) > 0.8
    assert np.amin(x_verts[:, 2]) < 1e-3


def test_errors():
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(
            0.0, 1.0, 0.0, 1.0, num_cells=(10, 10), out_file=tf.name
        )
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    with pytest.raises(ValueError):
        _ = StructuredQuadGrid(domain) # wrong cell type

    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_line([0., 0., 0.], [1., 0., 0.], num_cells=10, out_file=tf.name)
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=1)

    with pytest.raises(NotImplementedError):
        _ = StructuredQuadGrid(domain) # wrong tdim


@pytest.mark.parametrize("order,cell_type",[(1,"triangle"),(2,"triangle6")])
def test_fine_grid_creation(order, cell_type):
    # ### create coarse grid
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(
            0.0, 2.0, 0.0, 2.0, num_cells=(2, 2), recombine=True, out_file=tf.name
        )
        domain, _, _ = gmshio.read_from_msh(tf.name, MPI.COMM_WORLD, gdim=2)

    grid = StructuredQuadGrid(domain)

    # ### subdomain grid creation
    num_cells_subdomain = 10
    grid.fine_grid_method = [create_unit_cell_01]
    options = {"Mesh.ElementOrder": order}

    with tempfile.NamedTemporaryFile(suffix=".xdmf") as tf:
        with pytest.raises(NotImplementedError):
            grid.create_fine_grid(np.array([0, 1]), tf.name, "triangle9")

        grid.create_fine_grid(
            np.array([0, 1]), tf.name, cell_type, num_cells=num_cells_subdomain, options=options
        )
        with XDMFFile(MPI.COMM_WORLD, tf.name, "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
            ct = xdmf.read_meshtags(mesh, name="Grid")

        # remove the h5 as well
        h5 = pathlib.Path(tf.name).with_suffix(".h5")
        h5.unlink()

    assert ct.find(1).size > 0
    assert ct.find(2).size > 0

    with tempfile.NamedTemporaryFile(suffix=".xdmf") as tf:
        grid.create_fine_grid(
            np.array([0, ]), tf.name, cell_type, num_cells=num_cells_subdomain, options=options
        )
        with XDMFFile(MPI.COMM_WORLD, tf.name, "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
            ct = xdmf.read_meshtags(mesh, name="Grid")
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(tdim, fdim)
        fpath = pathlib.Path(tf.name)
        facets = fpath.parent / (fpath.stem + "_facets.xdmf")
        assert facets.exists()
        with XDMFFile(MPI.COMM_WORLD, facets.as_posix(), "r") as xdmf:
            ft = xdmf.read_meshtags(mesh, name="Grid")

        # remove the h5 as well
        h5 = pathlib.Path(tf.name).with_suffix(".h5")
        h5.unlink()

        # clean up the facet files
        facets.with_suffix(".h5").unlink()
        facets.unlink()

    assert ct.find(1).size > 0
    assert ct.find(2).size > 0
    assert ft.find(1).size == num_cells_subdomain
    assert ft.find(2).size == num_cells_subdomain
    assert ft.find(3).size == num_cells_subdomain
    assert ft.find(4).size == num_cells_subdomain
