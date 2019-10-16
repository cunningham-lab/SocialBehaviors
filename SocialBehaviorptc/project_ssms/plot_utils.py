import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.colors
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import joblib
import click
import os

from project_ssms.utils import downsample


def add_grid(x_grids, y_grids, grid_alpha=1.0):
    if x_grids is None or y_grids is None:
        return
    if isinstance(x_grids, torch.Tensor):
        x_grids = x_grids.numpy()
    if isinstance(y_grids, torch.Tensor):
        y_grids = y_grids.numpy()

    plt.scatter([x_grids[0], x_grids[0], x_grids[-1], x_grids[-1]],
                [y_grids[0], y_grids[1], y_grids[0], y_grids[1]], alpha=grid_alpha)
    for j in range(len(y_grids)):
        plt.plot([x_grids[0], x_grids[-1]], [y_grids[j], y_grids[j]], '--', color='grey', alpha=grid_alpha)

    for i in range(len(x_grids)):
        plt.plot([x_grids[i], x_grids[i]], [y_grids[0], y_grids[-1]], '--', color='grey', alpha=grid_alpha)


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


@click.command()
@click.option("--save_video_dir", default=None, help='name of the video')
@click.option("--checkpoint_dir", default=None, help='checkpoint dir')
@click.option("--video_t", default=None, help='making videos for the first video_T datapoints')
@click.option("--downsample_n", default=1, help='downsampling data[:video_T] by downsample_n times')
@click.option("--fps", default=25, help="fps, frame per second")
@click.option("--xlim", default=None, help='limit of the xrange')
@click.option("--ylim", default=None, help='limit of the yrange')
@click.option("--grid_alpha", default=0.8, help='alpha for the grid line')
def plot_animation(save_video_dir, checkpoint_dir, video_t, downsample_n, fps, xlim, ylim, grid_alpha):
    if save_video_dir is None:
        raise ValueError("Please provide save_video_dir!")
    if not os.path.exists(save_video_dir):
        os.makedirs(save_video_dir)

    video_T = int(video_t)

    model = joblib.load(checkpoint_dir + "/model")
    numbers = joblib.load(checkpoint_dir + "/numbers")

    try:
        x_grids = model.observation.transformation.x_grids
        y_grids = model.observation.transformation.y_grids
    except:
        x_grids = None
        y_grids = None

    K = model.K
    sample_x = numbers['sample_x']
    sample_z = numbers['sample_z']

    T, D = sample_x.shape
    if video_T is None:
        video_T = T

    quiver_args = {}
    # cluster_centers = None

    sample_x = downsample(sample_x[:video_T], downsample_n)
    sample_z = downsample(sample_z[:video_T], downsample_n)

    video_T, _ = sample_x.shape
    duration = int(video_T / fps)

    start = sample_x[:-1]
    end = sample_x[1:]
    dXY = end - start

    h = 1 / K
    ticks = [(1 / 2 + k) * h for k in range(K)]
    colors, cm = get_colors_and_cmap(K)

    # make animation 0
    fig = plt.figure(figsize=(16, 7))
    plt.axis('equal')

    plt.subplot(1, 2, 1)
    plt.quiver(start[:0, 0], start[:0, 1], dXY[:0, 0], dXY[:0, 1],
               angles='xy', scale_units='xy', scale=1, cmap=cm, color=colors[sample_z])
    cb = plt.colorbar(label='k', ticks=ticks)
    cb.set_ticklabels(range(K))
    add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
    plt.title("virgin")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.subplot(1, 2, 2)
    plt.quiver(start[:0, 2], start[:0, 3], dXY[:0, 2], dXY[:0, 3],
               angles='xy', scale_units='xy', scale=1, cmap=cm, color=colors[sample_z])
    cb = plt.colorbar(label='k', ticks=ticks)
    cb.set_ticklabels(range(K))
    add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
    plt.title("mother")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    def make_frame(t):
        num_frame = int(t * fps)
        plt.subplot(1, 2, 1)
        plt.quiver(start[:num_frame, 0], start[:num_frame, 1], dXY[:num_frame, 0], dXY[:num_frame, 1],
                   angles='xy', scale_units='xy', scale=1, cmap=cm, color=colors[sample_z])

        plt.subplot(1, 2, 2)
        plt.quiver(start[:num_frame, 2], start[:num_frame, 3], dXY[:num_frame, 2], dXY[:num_frame, 3],
                   angles='xy', scale_units='xy', scale=1, cmap=cm, color=colors[sample_z])

        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)
    animation.write_videofile("{}/ani_0.mp4".format(save_video_dir), fps=fps)

    # make animation 1
    fig = plt.figure(figsize=(7, 7))
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.axis('equal')
    add_grid(x_grids, y_grids, grid_alpha=grid_alpha)

    plt.quiver(start[:0, 0], start[:0, 1], dXY[:0, 0], dXY[:0, 1],
               angles='xy', scale_units='xy', scale=1, label="virgin", color='C0')
    plt.quiver(start[:0, 2], start[:0, 3], dXY[:0, 2], dXY[:0, 3],
               angles='xy', scale_units='xy', scale=1, label="mother", color='C1')
    #plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(0.5, 1.0))
    add_grid(x_grids, y_grids, grid_alpha=grid_alpha)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    def make_frame(t):
        num_frame = int(t * fps)

        plt.quiver(start[:num_frame, 0], start[:num_frame, 1], dXY[:num_frame, 0], dXY[:num_frame, 1],
                   angles='xy', scale_units='xy', scale=1, color='C0')
        plt.quiver(start[:num_frame, 2], start[:num_frame, 3], dXY[:num_frame, 2], dXY[:num_frame, 3],
                   angles='xy', scale_units='xy', scale=1, color='C1')

        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)
    animation.write_videofile("{}/ani_1.mp4".format(save_video_dir), fps=fps)


if __name__ == "__main__":
    plot_animation()




