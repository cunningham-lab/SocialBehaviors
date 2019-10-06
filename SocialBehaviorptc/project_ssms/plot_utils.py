import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.colors


def get_colors_and_cmap(K):
    if K <= 10:
        colors = np.array(plt.cm.tab10.colors)[:K]
        colors_list = colors.tolist()
    elif K <= 20:
        colors = np.array(plt.cm.tab20.colors)[:K]
        colors_list = colors.tolist()
    else:
        vals = np.linspace(0, 1, 10 * K)[10 * np.arange(K)]
        colors = plt.cm.tab20(vals)
        colors_list = colors.tolist()

    cvals = np.arange(K)

    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors_list))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples, K)

    return colors, cmap


def plot_z(z, K, plot_range=None, ylim=None, title=None):
    _, cmap = get_colors_and_cmap(K)
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

    h = (K-1) / K
    ticks = [(1 / 2 + k) * h for k in range(K)]
    cb = plt.colorbar(label='k', ticks=ticks)
    cb.set_ticklabels(range(K))


def plot_mouse(data, alpha=.8, title=None, xlim=None, ylim=None, mouse='both'):
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if title is not None:
        plt.title(title)
    _, D = data.shape
    assert D == 4 or D == 2
    if D == 4:
        plt.plot(data[:, 0], data[:, 1], label='virgin', alpha=alpha)
        plt.plot(data[:, 2], data[:, 3], label='mother', alpha=alpha)
    else:
        plt.plot(data[:, 0], data[:, 1], alpha=alpha, label=mouse)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)


def plot_4_traces(data, title):
    if title is not None:
        plt.title(title)
    plt.plot(data[:, 0], label='x1')
    plt.plot(data[:, 1], label='y1')
    plt.plot(data[:, 2], label='x2')
    plt.plot(data[:, 3], label='y2')
    plt.legend()

