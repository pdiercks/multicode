"""
make a contour plot using paraview in batch mode

Usage:
    pvbatch pv_contour_plot.py N field component PNG

Arguments:
    N              Number of Subdomains, i.e. XDMFFiles.
    field          Field variable to show.
    component      Component of the field variable.
    PNG            The target png filepath.

Options:
    -h, --help     Show this message and exit.
"""

import os
import sys
import argparse
from pathlib import Path
from paraview.simple import (
    ColorBy,
    GetActiveViewOrCreate,
    GetColorTransferFunction,
    GetDisplayProperties,
    GetSources,
    UpdatePipeline,
    SaveScreenshot,
    Show,
    XDMFReader,
)


def read_xdmf_range(xdmf_file, field, component):
    """read xdmf file and return data range"""

    source = XDMFReader(FileNames=[xdmf_file])
    UpdatePipeline()

    if field not in source.PointData.keys():
        raise ValueError(
            f"Field {field} does not exist. Choose one of {source.PointData.keys()}."
        )

    if component in (0, "x", "X"):
        this_range = 0
    elif component in (1, "y", "Y"):
        this_range = 1
    elif component in (2, "z", "Z"):
        this_range = 2
    else:
        # component = "Magnitude"
        this_range = -1

    array_info = source.PointData[field]
    return array_info.GetRange(this_range)


def trimit(png):
    args = ["convert", "-trim", png, png]
    os.system(" ".join(args))


def main(args):
    path = Path(args.Path)
    field = args.field
    component = args.component
    min_values = []
    max_values = []

    for i in range(args.N):
        source = path.parent / (path.stem + f"_{i}" + path.suffix)
        data_range = read_xdmf_range(source.as_posix(), field, component)
        min_values.append(data_range[0])
        max_values.append(data_range[1])

    Range = (min(min_values), max(max_values))
    sources = GetSources()
    view = GetActiveViewOrCreate("RenderView")
    for source in sources.values():
        display = GetDisplayProperties(source, view)
        ColorBy(display, ("POINTS", field, component))
        Show(source, view)

    # TODO display.SetScalarBarVisibility --> optional legend
    # display.SetScalarBarVisibility(view, True)
    fun = GetColorTransferFunction(field)
    fun.RescaleTransferFunction(Range[0], Range[1])

    # make all subdomains visible
    view.ResetCamera()
    # hide the coordinate system
    view.OrientationAxesVisibility = 0

    if args.viewsize is not None:
        view_size = args.viewsize
    else:
        view_size = view.ViewSize

    # save screenshot
    target = args.PNG
    SaveScreenshot(
        target,
        view,
        ImageResolution=view_size,
        FontScaling="Scale fonts proportionally",
        OverrideColorPalette="",
        StereoMode="No change",
        TransparentBackground=0,
        ImageQuality=95,
    )
    trimit(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=f"pvbatch {__file__}",
        description="Make a contour plot.",
        usage="%(prog)s [options] Path N field component PNG",
    )
    parser.add_argument("Path", type=str, help="FilePath to XDMFFiles to read.")
    parser.add_argument("N", type=int, help="Number of Subdomains, i.e. XDMFFiles.")
    parser.add_argument("field", type=str, help="Field variable to show")
    parser.add_argument("component", type=str, help="Component of the field variable")
    parser.add_argument("PNG", type=str, help="The target png filepath.")
    parser.add_argument(
        "--viewsize", nargs="+", type=int, help="Set the view size for higher quality."
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
