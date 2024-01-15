import matplotlib.pyplot as plt
import numpy as np


def plot_modes(edge_space, edge, modes, component, mask):
    """for hacking"""
    L = edge_space
    x_dofs = L.tabulate_dof_coordinates()

    if component in ("x", 0):
        modes = modes[:, ::2]
    elif component in ("y", 1):
        modes = modes[:, 1::2]

    assert edge in ("b", "bottom", "r", "right", "t", "top", "l", "left")
    xx = x_dofs[:, 0] # b or t
    if edge in ("r", "l"):
        xx = x_dofs[:, 1]
    oo = np.argsort(xx)

    for mode in modes[mask]:
        plt.plot(xx[oo], mode[oo])
    plt.show()



