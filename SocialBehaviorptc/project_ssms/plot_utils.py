import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

x,y,c = zip(*np.random.rand(30,3)*4)

default_colors = ["C0","C1","C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
# C0: blue, C1: orange, C2: green, C3: red, C4: purple, C5: brown, C6: pink, C7: grey, C8: yellow, C9: cyan


def get_camp(K):
    cvals = np.arange(K)
    colors = default_colors[:K]

    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    return cmap


def plot_z(z, K, plot_range=None, ylim=None, title=None):
    cmap = get_camp(K)
    if plot_range is None:
        plot_range=(0, z.shape[0])
    if ylim is None:
        ylim = (0, 1)
    plt.figure(figsize=(20, 2))
    if title is not None:
        plt.title(title)
    plt.imshow(z[None, plot_range[0]:plot_range[1]], aspect="auto", cmap=cmap,
               vmin=0, vmax=K-1, extent=(plot_range[0], plot_range[1], ylim[0], ylim[1]))
    plt.xlim(plot_range[0], plot_range[1])
    plt.ylabel("$z_{\\mathrm{true}}$")
    plt.yticks([])


def plot_1_mice(data, alpha=0.8, label='virgin'):
    plt.plot(data[:, 0], data[:, 1], label=label, alpha=alpha)


def plot_2_mice(data, alpha=0.8, title=None):
    if title is not None:
        plt.title(title)
    plt.plot(data[:,0], data[:,1], label='virgin', alpha=alpha)
    plt.plot(data[:,2], data[:,3], label='mother', alpha=alpha)
    #plt.legend()


def plot_4_traces(data, title):
    if title is not None:
        plt.title(title)
    plt.plot(data[:, 0], label='x1')
    plt.plot(data[:, 1], label='y1')
    plt.plot(data[:, 2], label='x2')
    plt.plot(data[:, 3], label='y2')
    plt.legend()


def plot_quiver(XYs, dXYs, mouse, other_mouse_loc=None, scale=1, alpha=1, title=None):
    if mouse == 'virgin':
        i = 0
        j = 1
    elif mouse == 'mother':
        i = 2
        j = 3

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    if title is not None:
        plt.suptitle(title)

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
