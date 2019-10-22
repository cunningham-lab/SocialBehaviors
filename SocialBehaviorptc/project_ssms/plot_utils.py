import numpy as np
import torch

import matplotlib.colors
from matplotlib import pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import joblib
import click
import os
import git

from project_ssms.utils import downsample
from ssm_ptc.utils import get_np

def add_grid(x_grids, y_grids, grid_alpha=1.0):
    if x_grids is None or y_grids is None:
        return
    if isinstance(x_grids, torch.Tensor):
        x_grids = get_np(x_grids)
    if isinstance(y_grids, torch.Tensor):
        y_grids = get_np(y_grids)

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
        data = get_np(data)
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
@click.option("--v_start", default=0, help='start of the true data')
@click.option("--v_end", default=5, help='end of the true data')
@click.option("--data_downsample_n", default=2, help="downsample times for real data")
@click.option("--checkpoint_dir", default=None, help='checkpoint dir')
@click.option("--video_downsample_n", default=1, help='downsampling data by downsample_n times')
@click.option("--fps", default=25, help="fps, frame per second")
@click.option("--xlim", default=None, help='limit of the xrange')
@click.option("--ylim", default=None, help='limit of the yrange')
@click.option("--grid_alpha", default=0.8, help='alpha for the grid line')
@click.option("--video_start_and_end", default="0,0.01", help='')
@click.option("--video_data_type", default="data", help="choose from 'data' or 'sample_x")
@click.option("--color_only", is_flag=True, help="")
@click.option("--traj_only", is_flag=True, help="")
def plot_animation_helper(save_video_dir, v_start, v_end, data_downsample_n, checkpoint_dir,
                          video_start_and_end, video_downsample_n, fps, xlim, ylim, grid_alpha, video_data_type,
                          color_only, traj_only):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if save_video_dir is None:
        raise ValueError("Please provide save_video_dir!")
    if not os.path.exists(save_video_dir):
        os.makedirs(save_video_dir)

    video_start, video_end = [float(v) for v in video_start_and_end.split(",")]

    model = joblib.load(checkpoint_dir + "/model")
    numbers = joblib.load(checkpoint_dir + "/numbers")

    try:
        x_grids = model.observation.transformation.x_grids
        y_grids = model.observation.transformation.y_grids
    except:
        x_grids = None
        y_grids = None

    K = model.K

    # data
    repo = git.Repo('.', search_parent_directories=True)  # SocialBehaviorectories=True)
    repo_dir = repo.working_tree_dir  # SocialBehavior

    if video_data_type == "data":
        data_dir = repo_dir + '/SocialBehaviorptc/data/trajs_all'
        trajs = joblib.load(data_dir)

        traj = trajs[36000 * v_start:36000 * v_end]
        traj = downsample(traj, data_downsample_n)
        data = torch.tensor(traj, dtype=torch.float64, device=device)
        z = numbers['z']

        T, D = data.shape
        start = int(T*video_start)
        end = int(T*video_end)

        data = downsample(data[start:end], video_downsample_n)
        z = downsample(z[start:end], video_downsample_n)
        plot_animation(save_video_dir, "data", data, z, K, fps, xlim, ylim, grid_alpha, x_grids, y_grids,
                       color_only, traj_only)

    elif video_data_type == "sample_x":
        # sample x
        sample_x = numbers['sample_x']
        sample_z = numbers['sample_z']

        T, D = sample_x.shape
        start = int(T*video_start)
        end = int(T*video_end)

        sample_x = downsample(sample_x[start:end], video_downsample_n)
        sample_z = downsample(sample_z[start:end], video_downsample_n)

        plot_animation(save_video_dir, "sample_x", sample_x, sample_z, K, fps, xlim, ylim, grid_alpha, x_grids, y_grids,
                       color_only, traj_only)


def plot_animation(save_video_dir, sample_name, x, z, K, fps, xlim, ylim, grid_alpha, x_grids, y_grids,
                   color_only=False, traj_only=False):


    video_T, _ = x.shape
    duration = int(video_T / fps)

    start = x[:-1]
    end = x[1:]
    dXY = end - start

    h = 1 / K
    ticks = [(1 / 2 + k) * h for k in range(K)]
    colors, cm = get_colors_and_cmap(K)

    if not traj_only:
        # make animation 0
        fig = plt.figure(figsize=(16, 7))
        plt.axis('equal')

        plt.subplot(1, 2, 1)
        plt.quiver(start[:0, 0], start[:0, 1], dXY[:0, 0], dXY[:0, 1],
                   angles='xy', scale_units='xy', scale=1, cmap=cm, color=colors[z])
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
                   angles='xy', scale_units='xy', scale=1, cmap=cm, color=colors[z])
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
            start_idx = max(0, num_frame-1)
            plt.quiver(start[start_idx:num_frame, 0], start[start_idx:num_frame, 1],
                       dXY[start_idx:num_frame, 0], dXY[start_idx:num_frame, 1],
                       angles='xy', scale_units='xy', scale=1, color=colors[z[start_idx]], cmap=cm)

            plt.subplot(1, 2, 2)
            plt.quiver(start[start_idx:num_frame, 2], start[start_idx:num_frame, 3],
                       dXY[start_idx:num_frame, 2], dXY[start_idx:num_frame, 3],
                       angles='xy', scale_units='xy', scale=1, color=colors[z[start_idx]], cmap=cm)

            return mplfig_to_npimage(fig)

        animation = VideoClip(make_frame, duration=duration)
        animation.write_videofile("{}/{}_colorful.mp4".format(save_video_dir, sample_name), fps=fps)

        #plt.quiver(start[:, 0], start[:, 1],dXY[:, 0], dXY[:, 1],
         #          angles='xy', scale_units='xy', scale=1, color=colors[z], cmap=cm)
        #plt.show()

    if not color_only:
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
            start_idx = max(0, num_frame-1)
            plt.quiver(start[start_idx:num_frame, 0], start[start_idx:num_frame, 1],
                       dXY[start_idx:num_frame, 0], dXY[start_idx:num_frame, 1],
                       angles='xy', scale_units='xy', scale=1, color='C0')
            plt.quiver(start[start_idx:num_frame, 2], start[start_idx:num_frame, 3],
                       dXY[start_idx:num_frame, 2], dXY[start_idx:num_frame, 3],
                       angles='xy', scale_units='xy', scale=1, color='C1')

            return mplfig_to_npimage(fig)

        animation = VideoClip(make_frame, duration=duration)
        animation.write_videofile("{}/{}_traj.mp4".format(save_video_dir, sample_name), fps=fps)


def plot_data_condition_on_zk(data, z, k, size=2, alpha=0.3):

    data_zk = data[z == k]
    plt.scatter(data_zk[:, 0], data_zk[:, 1], label='virgin', s=size, alpha=alpha)
    plt.scatter(data_zk[:, 2], data_zk[:, 3], label='mother', s=size, alpha=alpha)

    lgnd = plt.legend(loc='upper right', scatterpoints=1)
    for i in range(2):
        lgnd.legendHandles[i]._sizes = [30]


def plot_data_condition_on_all_zs(data, z, K, size=2, alpha=0.3):
    data = get_np(data)

    n_col = 5
    n_row = int(K/n_col)
    if K % n_col > 0:
        n_row += 1

    plt.figure(figsize=(20, 4*n_row))

    title = "spatial occupation under different hidden states"
    if title is not None:
        plt.suptitle(title)

    for k in range(K):
        plt.subplot(n_row, n_col, k+1)
        plot_data_condition_on_zk(data, z, k, size=size, alpha=alpha)
        plt.title('K={} '.format(k))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_2d_time_plot_condition_on_z(data, z, k, time_start, time_end, size=0.5):
    """
    assume data is np.ndarray, and time_start and time_end is not None
    """
    data = data[time_start:time_end]
    z = z[time_start:time_end]
    time_zk = np.where(z == k)[0]
    data_zk = data[z == k]

    plt.scatter(time_zk, data_zk[:, 0], label='virgin x', s=size)
    plt.scatter(time_zk, data_zk[:, 1], label='virgin y', s=size)

    plt.scatter(time_zk, data_zk[:, 2], label='mother x', s=size)
    plt.scatter(time_zk, data_zk[:, 3], label='mother y', s=size)

    plt.title('k={}'.format(k), x=-0.03, y=0.3, fontsize=20)

    lgnd = plt.legend(loc='upper right', scatterpoints=1)
    for i in range(4):
        lgnd.legendHandles[i]._sizes = [30]


def plot_2d_time_plot_condition_on_all_zs(data, z, K, title, time_start=None, time_end=None, size=0.5):
    data = get_np(data)

    T, _ = data.shape
    time_start = time_start if time_start else 0
    time_end = time_end if time_end else T

    plt.figure(figsize=(30, 2 * K))
    if title is not None:
        plt.suptitle(title)

    for k in range(K):
        plt.subplot(K, 1, k + 1)
        plot_2d_time_plot_condition_on_z(data, z, k, time_start, time_end, size)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


if __name__ == "__main__":
    plot_animation_helper()


