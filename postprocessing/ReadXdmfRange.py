import sys
import argparse
from paraview.simple import (
    ColorBy,
    GetActiveViewOrCreate,
    GetColorTransferFunction,
    GetDisplayProperties,
    GetSources,
    UpdatePipeline,
    Show,
    XDMFReader,
)


def read_xdmf_range(xdmf_file, field, component):
    """read xdmf file and return data range"""

    source = XDMFReader(FileNames=[xdmf_file])
    UpdatePipeline()

    if not field in source.PointData.keys():
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


def main(args):
    field = args.field
    component = args.component
    min_values = []
    max_values = []

    for i in range(args.N):
        source_base = f"/home/pdiercks/Repos/bam/2020_02_multiscale/lpanel/results/xdmf/rom_{i}.xdmf"
        data_range = read_xdmf_range(source_base, field, component)
        min_values.append(data_range[0])
        max_values.append(data_range[1])

    Range = (min(min_values), max(max_values))
    sources = GetSources()
    view = GetActiveViewOrCreate("RenderView")
    for source in sources.values():
        display = GetDisplayProperties(source, view)
        ColorBy(display, ("POINTS", field, component))
        Show(source, view)

    display.SetScalarBarVisibility(view, True)
    fun = GetColorTransferFunction(field)
    fun.RescaleTransferFunction(Range[0], Range[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read Subdomains.")
    parser.add_argument("N", type=int, help="Number of Subdomains.")
    parser.add_argument("field", type=str, help="Field variable to show")
    parser.add_argument("component", type=str, help="Component of the field variable")
    args = parser.parse_args(sys.argv[1:])
    main(args)
