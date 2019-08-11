import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hips.plotting.colormaps import gradient_cmap, white_to_color_cmap
color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange"
    ]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)


def plot_z(z, plot_range=None, ylim=None):
    if plot_range is None:
        plot_range=(0, z.shape[0])
    if ylim is None:
        ylim = (0, 1)
    plt.figure(figsize=(20, 2))
    plt.imshow(z[None, plot_range[0]:plot_range[1]], aspect="auto", cmap=cmap,
               vmin=0, vmax=len(colors) - 1, extent=(plot_range[0], plot_range[1], ylim[0], ylim[1]))
    plt.xlim(plot_range[0], plot_range[1])
    plt.ylabel("$z_{\\mathrm{true}}$")
    plt.yticks([])


def plot_2_mice(data, alpha):
    plt.plot(data[:,0], data[:,1], label='virgin', alpha=alpha)
    plt.plot(data[:,2], data[:,3], label='mother', alpha=alpha)
    plt.legend()


def plot_4_traces(data):
    plt.plot(data[:, 0], label='x1')
    plt.plot(data[:, 1], label='y1')
    plt.plot(data[:, 2], label='x2')
    plt.plot(data[:, 3], label='y2')
    plt.legend()


def plot_quiver(XYs, dXYs, mouse, other_mouse_loc=None, scale=1, alpha=1):
    if mouse == 'virgin':
        i = 0
        j = 1
    elif mouse == 'mother':
        i = 2
        j = 3

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0][0].quiver(XYs[:, i], XYs[:, j], dXYs[:, 0, i], dXYs[:, 0, j],
                     angles='xy', scale_units='xy', scale=scale, alpha=alpha)
    axs[0][0].set_title('K=0')

    axs[0][1].quiver(XYs[:, i], XYs[:, j], dXYs[:, 1, i], dXYs[:, 1, j],
                     angles='xy', scale_units='xy', scale=scale, alpha=alpha)
    axs[0][1].set_title('K=1')

    axs[1][0].quiver(XYs[:, i], XYs[:, j], dXYs[:, 2, i], dXYs[:, 2, j],
                     angles='xy', scale_units='xy', scale=scale, alpha=alpha)
    axs[1][0].set_title('K=2')

    axs[1][1].quiver(XYs[:, i], XYs[:, j], dXYs[:, 3, i], dXYs[:, 3, j],
                     angles='xy', scale_units='xy', scale=scale, alpha=alpha)
    axs[1][1].set_title('K=3')

    if other_mouse_loc is not None:
        axs[0][0].plot(*other_mouse_loc, 'ro')
        axs[0][1].plot(*other_mouse_loc, 'ro')
        axs[1][0].plot(*other_mouse_loc, 'ro')
        axs[1][1].plot(*other_mouse_loc, 'ro')
    """
    for row_axs in axs:
        for ax in row_axs:
            ax.set_xlim([0, 330])
            ax.set_ylim([0, 380])
    """
    plt.tight_layout()
