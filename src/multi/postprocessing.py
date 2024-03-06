import yaml
import numpy as np
from typing import Optional
from pathlib import Path

import pyvista
from dolfinx import mesh, plot
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


def plot_domain(domain: mesh.Mesh, cell_tags: Optional[mesh.MeshTags] = None, colormap: str = "viridis", transparent: bool = False, output: Optional[str] = None):

    if colormap == "bam-RdBu":
        from multi.postprocessing import read_bam_colormap

        bam_RdBu = read_bam_colormap()
        cmap = ListedColormap(bam_RdBu, name="bam-RdBu")
    elif colormap == "RdYlBu":
        cmap = "RdYlBu"
    else:
        cmap = "viridis"

    if output is not None:
        off_screen = True
        pyvista.start_xvfb(wait=0)
    else:
        off_screen = False


    tdim = domain.topology.dim
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    if cell_tags is not None:
        grid.cell_data["Marker"] = cell_tags.values
        grid.set_active_scalars("Marker")

    plotter = pyvista.Plotter(off_screen=off_screen)
    plotter.add_mesh(grid, show_edges=True, show_scalar_bar=False, cmap=cmap)
    pyvista.global_theme.font.size = 12
    plotter.show_bounds(
        color="black",
        font_family="times",
        location="outer",
        xtitle="X",
        ytitle="Y",
        use_2d=False,
        bold=True,
        ticks="outside",
    )
    plotter.view_xy()

    if off_screen:
        plotter.screenshot(output, transparent_background=transparent)
    else:
        plotter.show()

