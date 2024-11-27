import yaml
import numpy as np
from typing import Optional
from pathlib import Path

import pyvista
import dolfinx as df
import matplotlib as mpl
from matplotlib.colors import ListedColormap


def read_bam_colors():
    """get primary(0), secondary(1), tertiary(2) or all(3) for
    bam colors black, blue, green red, and yellow"""
    yamlfile = Path(__file__).parent / "bamcolors_rgb.yml"
    with yamlfile.open("r") as instream:
        bam_cd = yaml.safe_load(instream)
    return bam_cd


def read_bam_colormap():
    infile = Path(__file__).parent / "bam-RdBu.npy"
    cmap = np.load(infile.as_posix())
    return cmap


def plot_domain(
    domain: df.mesh.Mesh,
    cell_tags: Optional[df.mesh.MeshTags] = None,
    colormap: str = "viridis",
    transparent: bool = False,
    output: Optional[str] = None,
):
    """Visualize the ``domain`` using pyvista.

    Args:
        domain: The ``dolfinx.mesh.Mesh`` to visualize.
        cell_tags: If not None, show cell tags.
        colormap: The colormap to be used. Can be one of 'viridis', 'RdYlBu', 'bam-RdBu'.
        transparent: If True, use a transparent background.
        output: If not None, save screenshot to this filepath.

    """

    if colormap == "bam-RdBu":
        from multi.postprocessing import read_bam_colormap

        bam_RdBu = read_bam_colormap()
        cmap = ListedColormap(bam_RdBu, name="bam-RdBu")
    else:
        cmap = colormap

    if output is not None:
        off_screen = True
        pyvista.start_xvfb(wait=0)
    else:
        off_screen = False

    tdim = domain.topology.dim
    topology, cell_types, geometry = df.plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    if cell_tags is not None:
        grid.cell_data["Marker"] = cell_tags.values
        grid.set_active_scalars("Marker")

    plotter = pyvista.Plotter(off_screen=off_screen)
    plotter.add_mesh(grid, show_edges=True, show_scalar_bar=False, cmap=cmap)
    plotter.view_xy()

    if off_screen:
        plotter.screenshot(output, transparent_background=transparent)
    else:
        plotter.show()


def plot_scalar_function(
    f: df.fem.Function,
    pngfile: str,
    axes: bool = False,
    warp_by: float = 1.0,
    add_grid: bool = False,
    cmap: Optional[str] = None,
):
    """Plot a scalar function using pyvista.

    Args:
        f: The Function to plot.
        pngfile: Filepath for screenshot.
        axes: Whether coordinate system is plotted.
        warp_by: Factor for `warp_by_scalar`.
        add_grid: Whether the mesh is added to the plot.
        cmap: Matplotlib colormap.
    """
    if cmap is None:
        colormap = mpl.colormaps.get_cmap("viridis").resampled(25)
    else:
        colormap = mpl.colormaps.get_cmap(cmap).resampled(25)

    plotter = pyvista.Plotter(off_screen=True)
    V = f.function_space
    grid = pyvista.UnstructuredGrid(*df.plot.vtk_mesh(V))
    if add_grid:
        plotter.add_mesh(
            grid,
            show_edges=True,
            lighting=False,
            opacity=0.5,
            show_scalar_bar=False,
            color="lightblue",
        )

    grid.point_data[f.name] = f.x.array
    warped = grid.warp_by_scalar(f.name, factor=warp_by)
    plotter.add_mesh(
        warped,
        show_edges=True,
        lighting=False,
        cmap=colormap,
        clim=[0, max(f.x.array)],
        scalar_bar_args={"title": f.name},
        show_scalar_bar=False,
    )
    if axes:
        plotter.show_axes()
    plotter.link_views()
    plotter.view_isometric()
    plotter.camera.zoom(1.0)
    plotter.screenshot(pngfile, transparent_background=True)
