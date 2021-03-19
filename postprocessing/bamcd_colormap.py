import yaml
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

with open("bamcolors_hex.yml", "r") as f:
    bamcd = yaml.safe_load(f)


def hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) / 255 for i in range(0, lv, lv // 3))


def plot_colors(colors):
    for i, c in enumerate(colors):
        plt.plot(np.ones(5) * i, bamcd[c]["c"], label=c)
    plt.legend()
    plt.show()


colors_to_removed = [
    "BAMblue3",
    "BAMblue4",
    "BAMgrad010",
    "BAMgrad020",
    "BAMgrad050",
    "BAMgrad080",
    "BAMgrad100",
    "BAMgreen4",
    "BAMgreen2",
]

# plot_colors(["BAMblue1", "BAMblue2", "BAMblue3", "BAMblue4"])
# plot_colors(["BAMgreen1", "BAMgreen2", "BAMgreen3", "BAMgreen4"])
# breakpoint()

for k in colors_to_removed:
    bamcd.pop(k)

rgb = []
for k, v in bamcd.items():
    rgb.append(hex_to_rgb(v["c"]))


def plot_cm(rgb):
    x = np.arange(0, np.pi, 0.1)
    y = np.arange(0, 2 * np.pi, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(X) * np.sin(Y) * 10
    n_bins = [10, 100, 150, 255]
    cmap_name = "bam_cd"
    fig, axs = plt.subplots(2, 2, figsize=(6, 9))
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    for n_bin, ax in zip(n_bins, axs.ravel()):
        cm = LinearSegmentedColormap.from_list(cmap_name, rgb, N=n_bin)
        im = ax.imshow(Z, interpolation="nearest", origin="lower", cmap=cm)
        ax.set_title("N bins: %s" % n_bin)
        fig.colorbar(im, ax=ax)
    plt.show()


hsv = rgb.copy()
hsv.sort(key=lambda rgb: colorsys.rgb_to_hsv(*rgb))
plot_cm(hsv)

np.save("bamcm.npy", hsv)
