import yaml
import numpy as np
from pathlib import Path


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
