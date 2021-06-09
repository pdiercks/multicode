"""
Type
pvbath pv_legend.py --help
for help.
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
    Render,
    SaveScreenshot,
    UpdatePipeline,
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


def get_custom_labels(vmin, vmax, num):
    x = linspace(vmin, vmax, num=num)
    return x.tolist()


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
    view.ViewSize = args.viewsize

    sources = GetSources()
    for source in sources.values():
        display = GetDisplayProperties(source, view)
        ColorBy(display, ("POINTS", field, component))
        Hide(source, view)

    # set default
    view.InteractionMode = "2D"
    # FIXME messing with the camera might be necessary depending on args.length and args.pos
    camera = view.GetActiveCamera()
    camera.SetFocalPoint(10, 10, 0)
    camera.SetPosition(0, 0, 20)
    view.Update()
    # view.ResetCamera()

    ctf = GetColorTransferFunction(field)
    ctf.RescaleTransferFunction(Range[0], Range[1])
    ctf.RGBPoints = [Range[0], *BAM_BLUE, 0.0, *STRUCTURE, Range[1], *BAM_RED]

    fontsize = args.fontsize
    ColorBar = GetScalarBar(ctf, view)
    if args.orientation.lower() in ("horizontal",):
        ColorBar.Orientation = "Horizontal"
    else:
        ColorBar.Orientation = "Vertical"
    ColorBar.Title = rf"{args.title}"
    ColorBar.ComponentTitle = ""
    ColorBar.TitleColor = BLACK
    ColorBar.LabelColor = BLACK
    ColorBar.TitleFontSize = fontsize
    ColorBar.LabelFontSize = int(fontsize * 0.95)
    ColorBar.ScalarBarThickness = args.thickness

    # change scalar bar placement
    ColorBar.WindowLocation = "AnyLocation"
    ColorBar.Position = args.pos
    ColorBar.ScalarBarLength = args.length

    ColorBar.AutomaticLabelFormat = 0
    ColorBar.UseCustomLabels = 1
    num_ticks = args.numticks
    ColorBar.CustomLabels = get_custom_labels(Range[0], Range[1], num_ticks)
    ColorBar.LabelFormat = "%#3.1e"
    ColorBar.RangeLabelFormat = "%#3.1e"
    ColorBar.TitleFontFamily = args.font
    ColorBar.LabelFontFamily = args.font

    # hide the coordinate system
    view.OrientationAxesVisibility = 0

    # make changes take effect
    Render(view)

    target = args.PNG
    SaveScreenshot(
        target,
        view,
        ImageResolution=view.ViewSize,
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
        usage="%(prog)s [options] Path field component PNG",
    )
    parser.add_argument("Path", type=str, help="FilePath to XDMFFile to read.")
    parser.add_argument("field", type=str, help="Field variable to show")
    parser.add_argument("component", type=str, help="Component of the field variable")
    parser.add_argument("PNG", type=str, help="The target png filepath.")
    parser.add_argument(
        "--viewsize",
        nargs="+",
        type=int,
        help="Set the view size for higher quality.",
        default=[1468, 678],
    )
    parser.add_argument(
        "--range",
        nargs="+",
        type=float,
        help="Set the value range (default is determined from Path).",
    )
    parser.add_argument(
        "--title", type=str, help="Set the title of the legend.", default="f"
    )
    parser.add_argument(
        "--orientation",
        type=str,
        help="Set orientation of the legend.",
        default="Vertical",
    )
    parser.add_argument("--numticks", type=int, help="Set number of ticks.", default=5)
    parser.add_argument(
        "--pos",
        type=float,
        nargs="+",
        default=[0.5, 0.5],
        help="Set position of the bar",
    )
    parser.add_argument(
        "--font", type=str, default="Arial", help="Set font family (default: Arial)"
    )
    parser.add_argument(
        "--fontsize", type=int, default=16, help="Set title font size (default: 16)"
    )
    parser.add_argument(
        "--length", type=float, default=0.3, help="Set length of the bar (default: 0.3)"
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=16,
        help="Set thickness of the bar (default: 16)",
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
