import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

with open("bamcolors_raw.yml", "r") as f:
    bamcd = yaml.safe_load(f)

red_colors = bamcd["red"]["rgb"]
blue_colors = bamcd["blue"]["rgb"]
# pick primary
red = red_colors[0]
blue = blue_colors[0]

# create colormaps for each color ranging from (RGB of) color to white (1)
# instead of white use lightest bam black (204, 216, 223)

N = 256
r = np.ones((N, 4))
r[:, 0] = np.linspace(red[0] / 255, 204 / 255, N)
r[:, 1] = np.linspace(red[1] / 255, 216 / 255, N)
r[:, 2] = np.linspace(red[2] / 255, 223 / 255, N)
r_cmp = ListedColormap(r)

b = np.ones((N, 4))
b[:, 0] = np.linspace(blue[0] / 255, 204 / 255, N)
b[:, 1] = np.linspace(blue[1] / 255, 216 / 255, N)
b[:, 2] = np.linspace(blue[2] / 255, 223 / 255, N)
b_cmp = ListedColormap(b)

# combine, revert the red colormap
new = np.vstack((b_cmp(np.linspace(0, 1, 128)), r_cmp(np.linspace(1, 0, 128))))
new_cmp = ListedColormap(new, name='bam-RdBu')  # named after diverging colormap in matplotlib
# RdYlBu is also a good one

plt.figure(figsize=(7, 6))
data = np.random.random([100, 100]) * 10
plt.pcolormesh(data, cmap=new_cmp)
plt.colorbar()

plt.show()

np.save("bam-RdBu.npy", new)
