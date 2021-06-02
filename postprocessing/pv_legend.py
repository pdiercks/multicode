"""
save legend to png

Usage:
    pvbatch pv_legend.py Path field component PNG

Arguments:
    field          Field variable to show.
    PNG            The target png filepath.

Options:
    -h, --help     Show this message and exit.
"""

import os
import sys
import argparse
from pathlib import Path
from numpy import linspace
from paraview.simple import (
    ColorBy,
    GetActiveViewOrCreate,
    GetColorTransferFunction,
    GetDisplayProperties,
    GetScalarBar,
    GetSources,
    Hide,
    UpdatePipeline,
    SaveScreenshot,
    XDMFReader,
)

BAM_BLUE = [0.0, 0.6863, 0.9412]
WHITE = [1.0, 1.0, 1.0]
BAM_RED = [0.8235, 0.0, 0.1176]
BLACK = [
    0.027450980392156862,
    0.050980392156862744,
    0.050980392156862744,
]  # bam primary black
BLACK_GRAD010 = [0.8, 0.8470588235294118, 0.8745098039215686]
# set color for value zero (the structure)
STRUCTURE = BLACK_GRAD010


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
    source = Path(args.Path)
    field = args.field
    component = args.component
    data_range = read_xdmf_range(source.as_posix(), field, component)

    if args.range is not None:
        Range = args.range
    else:
        Range = (data_range[0], data_range[1])

    view = GetActiveViewOrCreate("RenderView")
    sources = GetSources()
    for source in sources.values():
        display = GetDisplayProperties(source, view)
        ColorBy(display, ("POINTS", field, component))
        Hide(source, view)
        # Show(source, view)

    ctf = GetColorTransferFunction(field)
    ctf.RescaleTransferFunction(Range[0], Range[1])
    ctf.RGBPoints = [Range[0], *BAM_BLUE, 0.0, *STRUCTURE, Range[1], *BAM_RED]

    fontsize = 30
    ColorBar = GetScalarBar(ctf, view)
    ColorBar.AutoOrient = 0
    ColorBar.Orientation = "Horizontal"
    ColorBar.Title = "A"
    ColorBar.ComponentTitle = ""
    ColorBar.TitleColor = BLACK
    ColorBar.LabelColor = BLACK
    ColorBar.TitleFontSize = fontsize
    ColorBar.LabelFontSize = int(fontsize * 0.95)
    ColorBar.ScalarBarThickness = int(fontsize / 2.0)
    # ColorBar.ScalarBarLength = 0.5

    def get_custom_labels(vmin, vmax, num):
        x = linspace(vmin, vmax, num=num)
        return x.tolist()

    ColorBar.AutomaticLabelFormat = 0
    ColorBar.UseCustomLabels = 1
    num_ticks = 5
    ColorBar.CustomLabels = get_custom_labels(Range[0], Range[1], num_ticks)
    ColorBar.LabelFormat = "%#3.1e"
    ColorBar.RangeLabelFormat = "%#3.1e"
    ColorBar.TitleFontFamily = "Times"
    ColorBar.LabelFontFamily = "Times"

    # TODO display.SetScalarBarVisibility --> optional legend
    # display.SetScalarBarVisibility(view, True)
    # fun = GetColorTransferFunction(field)
    # fun.RescaleTransferFunction(Range[0], Range[1])

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
        description="Plot the legend for given data.",
        usage="%(prog)s [options] Path N field component PNG",
    )
    parser.add_argument("Path", type=str, help="FilePath to XDMFFile to read.")
    parser.add_argument("field", type=str, help="Field variable to show")
    parser.add_argument("component", type=str, help="Component of the field variable")
    parser.add_argument("PNG", type=str, help="The target png filepath.")
    parser.add_argument(
        "--viewsize", nargs="+", type=int, help="Set the view size for higher quality."
    )
    parser.add_argument("--range", nargs="+", type=float, help="Set the value range.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
