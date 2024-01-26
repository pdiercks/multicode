"""PlottingContext written by Christoph Pohl published here https://doi.org/10.5281/zenodo.4452800 as plots/plotstuff.py"""

# ### Matplotlib customization
# https://matplotlib.org/stable/users/explain/customizing.html#customizing-matplotlib-with-style-sheets-and-rcparams

from pathlib import Path
import subprocess
import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt


class PlottingContext:
    def __init__(self, argv: list[str], styles: list[str]):
        """Initializes a plotting context manager.

        Args:
            argv: List of command line arguments.
            styles: The style(s) to use. This can either be a list of known matplotlib
            styles or stylefiles to read from.
        """

        if len(argv) == 2:
            self.write_output = True
            self.filename = argv[1]
        else:
            self.write_output = False

        # styles are composable
        _styles = []
        for style in styles:
            if style in plt.style.available:
                _styles.append(style)
            else:
                path = Path(style)
                if path.exists():
                    _styles.append(str(path))
                else:
                    stylefile = Path(__file__).parent / (path.stem + ".mplstyle")
                    if stylefile.exists():
                        _styles.append(str(stylefile))
                    else:
                        raise FileNotFoundError(f"The file {str(stylefile)} could not be found.")
        plt.style.use(_styles)
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
