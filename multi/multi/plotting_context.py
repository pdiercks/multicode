"""PlottingContext written by Christoph Pohl
published here https://doi.org/10.5281/zenodo.4452800 as
plots/plotstuff.py"""
from pathlib import Path
import subprocess
import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt


class PlottingContext:
    def __init__(self, argv, style):
        if len(argv) == 2:
            self.write_output = True
            self.filename = argv[1]
        else:
            self.write_output = False
        if style in plt.style.available:
            plt.style.use(style)
        else:
            stylefile = Path(__file__).parent / (style + ".mplstyle")
            plt.style.use(str(stylefile))
        self.fig = plt.figure(constrained_layout=True)

    def __enter__(self):
        return self.fig

    def __exit__(self, *args):
        if self.write_output:
            plt.savefig(self.filename)
        else:
            plt.savefig("/tmp/current_mpl_plot.pdf")
            try:
                subprocess.run(
                    "ps aux | grep current_mpl_plot.pdf | grep -v grep",
                    shell=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                subprocess.run(["xdg-open", "/tmp/current_mpl_plot.pdf"])
