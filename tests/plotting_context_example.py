import sys
import numpy as np
from multi.plotting_context import PlottingContext


if __name__ == "__main__":
    with PlottingContext(sys.argv, ["paper_onecol"]) as fig:
        ax = fig.subplots()

        x = np.linspace(0, 1, num=11)
        y = np.sin(2 * np.pi * x)
        ax.plot(x, y, '-x')
        ax.plot(x, 2 * y, '-o')
