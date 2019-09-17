import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.colors

x,y,c = zip(*np.random.rand(30,3)*4)

default_colors = ["C0","C1","C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
# C0: blue, C1: orange, C2: green, C3: red, C4: purple, C5: brown, C6: pink, C7: grey, C8: yellow, C9: cyan


def get_cmap(K):
    cvals = np.arange(K)
    colors = default_colors[:K]

    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples, K)

    return cmap


def plot_z(z, K, plot_range=None, ylim=None, title=None):
    cmap = get_cmap(K)
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


def plot_1_mice(data, alpha=0.8, label='virgin'):
    plt.plot(data[:, 0], data[:, 1], label=label, alpha=alpha)


def plot_2_mice(data, alpha=0.8, title=None, xlim=None, ylim=None):
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if title is not None:
        plt.title(title)
    plt.plot(data[:,0], data[:,1], label='virgin', alpha=alpha)
    plt.plot(data[:,2], data[:,3], label='mother', alpha=alpha)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    #plt.legend()


def plot_4_traces(data, title):
    if title is not None:
        plt.title(title)
    plt.plot(data[:, 0], label='x1')
    plt.plot(data[:, 1], label='y1')
    plt.plot(data[:, 2], label='x2')
    plt.plot(data[:, 3], label='y2')
    plt.legend()
