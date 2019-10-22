import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation

import seaborn as sns
from torch import __init__

from ssm_ptc.utils import check_and_convert_to_tensor, get_np
from project_ssms.plot_utils import get_colors_and_cmap
from project_ssms.utils import downsample


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


def add_grid_to_ax(ax, x_grids, y_grids):
    if isinstance(x_grids, torch.Tensor):
        x_grids = get_np(x_grids)
    if isinstance(y_grids, torch.Tensor):
        y_grids = get_np(y_grids)

    ax.scatter([x_grids[0], x_grids[0], x_grids[-1], x_grids[-1]], [y_grids[0], y_grids[-1], y_grids[0], y_grids[-1]])
    for j in range(len(y_grids)):
        ax.plot([x_grids[0], x_grids[-1]], [y_grids[j], y_grids[j]], '--', color='grey')

    for i in range(len(x_grids)):
        ax.plot([x_grids[i], x_grids[i]], [y_grids[0], y_grids[-1]], '--', color='grey')


def plot_realdata_quiver(realdata, z, K, x_grids=None, y_grids=None,
                         xlim=None, ylim=None, title=None, cluster_centers=None, grid_alpha=0.8, **quiver_args):
    # TODO: fix the case for K=1. color mapping only work for K>=2. need to use only one color for K=1
    if isinstance(realdata, torch.Tensor):
        realdata = get_np(realdata)

    _, D = realdata.shape
    assert D == 4 or D == 2

    start = realdata[:-1]
    end = realdata[1:]
    dXY = end - start

    h = 1 / K
    ticks = [(1 / 2 + k) * h for k in range(K)]
    colors, cm = get_colors_and_cmap(K)

    if D == 2:
        plt.figure(figsize=(8, 7))
        if title is not None:
            plt.title(title)

        plt.quiver(start[:, 0], start[:, 1], dXY[:, 0], dXY[:, 1],
                   angles='xy', scale_units='xy', scale=1, cmap=cm, color=colors[z], **quiver_args)
        cb = plt.colorbar(label='k', ticks=ticks)
        cb.set_ticklabels(range(K))

        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        if cluster_centers is not None:
            assert isinstance(cluster_centers, np.ndarray), "cluster_centers should be ndarray."
            plt.scatter(cluster_centers[:,0], cluster_centers[:,1], color='k', marker='*')
    else:

        plt.figure(figsize=(16, 7))
        if title is not None:
            plt.suptitle(title)

        plt.subplot(1, 2, 1)
        plt.quiver(start[:, 0], start[:, 1], dXY[:, 0], dXY[:, 1],
               angles='xy', scale_units='xy', scale=1, cmap=cm, color=colors[z], **quiver_args)
        cb = plt.colorbar(label='k', ticks=ticks)
        cb.set_ticklabels(range(K))

        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
        plt.title("virgin")
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        if cluster_centers is not None:
            assert isinstance(cluster_centers, np.ndarray), "cluster_centers should be ndarray."
            plt.scatter(cluster_centers[:,0], cluster_centers[:,1], color='k', marker='*')

        plt.subplot(1, 2, 2)
        plt.quiver(start[:, 2], start[:, 3], dXY[:, 2], dXY[:, 3],
                   angles='xy', scale_units='xy', scale=1, cmap=cm, color=colors[z], **quiver_args)
        cb = plt.colorbar(label='k', ticks=ticks)
        cb.set_ticklabels(range(K))

        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
        plt.title("mother")
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        if cluster_centers is not None:
            assert isinstance(cluster_centers, np.ndarray), "cluster_centers should be ndarray."
            plt.scatter(cluster_centers[:,2], cluster_centers[:,3], color='k', marker='*')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_animation(x, z, K, mouse='both', x_grids=None, y_grids=None, grid_alpha=0.8, xlim=None, ylim=None,
                   video_name=None, interval=2, downsample_n=1, max_length=None):
    if video_name is None:
        video_name = "anim_" + mouse

    T, D = x.shape

    if max_length is not None:
        assert isinstance(max_length, int), "max_length must be int."
        downsample_n = int(T / max_length)

    x = downsample(x, downsample_n)
    z = downsample(z, downsample_n)

    colors, cm = get_colors_and_cmap(K)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure(figsize=(8, 7))
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.axis('equal')
    add_grid(x_grids, y_grids, grid_alpha=grid_alpha)

    if mouse == 'virgin' or mouse == 'mother':
        assert D == 2

        XY = x[:-1]
        dXY = x[1:] - x[:-1]
        Q = plt.quiver([], [], [], [], scale=1, units="xy", cmap=cm, color=colors[z])

        def update_quiver(num, Q, XY, dXY):
            """updates the horizontal and vertical vector components by a
            fixed increment on each frame
            """

            U = dXY[:num + 1, 0]
            V = dXY[:num + 1, 1]

            Q.set_offsets(XY[:num + 1])
            Q.set_UVC(U, V)
            return Q,
    else:
        assert mouse == "both" and D == 4
        XY = np.zeros((2 * (T - 1), 2))
        XY[2 * np.arange(T - 1)] = x[:-1, 0:2]
        XY[2 * np.arange(T - 1) + 1] = x[:-1, 2:4]

        diff = x[1:] - x[:-1]  # (T-1, 4)
        dXY = np.zeros((2 * (T - 1), 2))
        dXY[2 * np.arange(T - 1)] = diff[:, 0:2]
        dXY[2 * np.arange(T - 1) + 1] = diff[:, 2:4]

        two_z = np.repeat(z, 2)
        Q = plt.quiver([], [], [], [], scale=1, units="xy", cmap=cm, color=colors[two_z])

        def update_quiver(num, Q, XY, dXY):
            """updates the horizontal and vertical vector components by a
            fixed increment on each frame
            """

            U = dXY[:2 * (num + 1), 0]
            V = dXY[:2 * (num + 1), 1]

            Q.set_offsets(XY[:2 * (num + 1)])
            Q.set_UVC(U, V)
            return Q,

    anim = animation.FuncAnimation(fig, update_quiver, frames=T - 1, fargs=(Q, XY, dXY),
                                   interval=interval, blit=False)
    anim.save("{}.mp4".format(video_name), writer=writer)


def plot_cluster_centers(cluster_centers, x_grids, y_grids, grid_alpha=0.8):
    assert isinstance(cluster_centers, np.ndarray), "cluster_centers should be ndarray."

    K, D = cluster_centers.shape
    assert D == 2 or D == 4

    h = 1 / K
    ticks = [(1 / 2 + k) * h for k in range(K)]
    colors, cm = get_colors_and_cmap(K)

    xlim = (x_grids[0] - 20, x_grids[-1] + 20)
    ylim = (y_grids[0] - 20, y_grids[-1] + 20)

    if D == 2:
        plt.figure(figsize=(8, 7))

        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', color=colors, s=80)
        plt.quiver([xlim[0] - 50], [ylim[0] - 50], [-1, -1], [-1, -1], cmap=cm)
        cb = plt.colorbar(label='k', ticks=ticks)
        cb.set_ticklabels(range(K))

        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)

        plt.xlim(xlim)
        plt.ylim(ylim)

    else:

        plt.figure(figsize=(16, 7))

        plt.subplot(1, 2, 1)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', color=colors, s=80)
        plt.quiver([xlim[0] - 50], [ylim[0] - 50], [-1, -1], [-1, -1], cmap=cm)
        cb = plt.colorbar(label='k', ticks=ticks)
        cb.set_ticklabels(range(K))
        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
        plt.title("virgin")
        plt.xlim(xlim)
        plt.ylim(ylim)

        plt.subplot(1, 2, 2)
        plt.scatter(cluster_centers[:, 2], cluster_centers[:, 3], marker='*', color=colors, s=80)
        plt.quiver([xlim[0] - 50], [ylim[0] - 50], [-1, -1], [-1, -1], cmap=cm)
        cb = plt.colorbar(label='k', ticks=ticks)
        cb.set_ticklabels(range(K))
        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
        plt.title("mother")
        plt.xlim(xlim)
        plt.ylim(ylim)


def plot_weights(weights, Df, K, x_grids, y_grids, max_weight=10, title=None):
    assert Df == 4

    n_grid_x = len(x_grids) - 1
    n_grid_y = len(y_grids) - 1

    assert weights.shape == (n_grid_x*n_grid_y, K, Df)

    plt.figure(figsize=(n_grid_x*5, n_grid_y*4))
    if title is not None:
        plt.suptitle(title)

    gap = 0.8 / (K - 1)

    for j in range(n_grid_y):
        for i in range(n_grid_x):
            plot_idx = (n_grid_y - j - 1) * n_grid_x + i + 1
            plt.subplot(n_grid_y, n_grid_x, plot_idx)

            grid_idx = i * n_grid_y + j
            plt.title("Grid {}".format(grid_idx))

            for k in range(K):
                plt.bar(np.arange(Df) - 0.4 + k * gap, weights[grid_idx][k], width=.2, label='k={}'.format(k))
            plt.plot([0, Df], [0, 0], '-k')
            plt.ylim(0, max_weight)
            plt.xticks(np.arange(0, 4, 1), ["lower L", "upper L", "lower R", "upper R"])
            plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def add_percentage(k, percentage, grid_centers):
    if percentage is None:
        return

    G = len(percentage)

    texts = ["{0:.2f}%".format(percentage[g][k] * 100) for g in range(G)]

    for c, text in zip(grid_centers, texts):
        plt.text(c[0] + 20, c[1] + 40, text, fontsize=12, color='k')


def plot_dynamics(weighted_corner_vecs, animal, x_grids, y_grids, K, scale=0.1, percentage=None, title=None,
                  grid_alpha=1):
    """
    This is for the illustration of the dynamics of the discrete grid model. Probably want to make the case of K>8
    """
    if isinstance(x_grids, torch.Tensor):
        x_grids = get_np(x_grids)
    if isinstance(y_grids, torch.Tensor):
        y_grids = get_np(y_grids)

    result_corner_vecs = np.sum(weighted_corner_vecs, axis=2)
    n_x = len(x_grids) - 1
    n_y = len(y_grids) - 1
    grid_centers = np.array(
        [[1 / 2 * (x_grids[i] + x_grids[i + 1]), 1 / 2 * (y_grids[j] + y_grids[j + 1])] for i in range(n_x) for j in
         range(n_y)])
    Df = 4

    def plot_dynamics_k(k):
        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)

        add_percentage(k, percentage=percentage, grid_centers=grid_centers)

        for df in range(Df):
            plt.quiver(grid_centers[:, 0], grid_centers[:, 1], weighted_corner_vecs[:, k, df, 0],
                       weighted_corner_vecs[:, k, df, 1],
                       units='xy', scale=scale, width=2, alpha=0.5)

        plt.quiver(grid_centers[:, 0], grid_centers[:, 1], result_corner_vecs[:, k, 0], result_corner_vecs[:, k, 1],
                   units='xy', scale=scale, width=2, color='red', alpha=0.5)
        plt.title("K={}, ".format(k) + animal, fontsize=20)

    if K <= 4:
        plt.figure(figsize=(20, 5))
        if title is not None:
            plt.suptitle(title)
        for k in range(K):
            plt.subplot(1, K, k+1)
            plot_dynamics_k(k)
    elif 4 < K <= 8:
        plt.figure(figsize=(20, 10))
        if title is not None:
            plt.suptitle(title)
        for k in range(K):
            plt.subplot(2, int(K/2)+1, k+1)
            plot_dynamics_k(k)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_quiver(XYs, dXYs, mouse, K,scale=1, alpha=1, title=None, x_grids=None, y_grids=None, grid_alpha=1):

    n_row = int(K/5)
    if K % 5 > 0:
        n_row += 1

    plt.figure(figsize=(20, 4*n_row))
    if title is not None:
        plt.suptitle(title)

    for k in range(K):
        plt.subplot(n_row, 5, k+1)
        plt.quiver(XYs[:, 0], XYs[:, 1], dXYs[:, k, 0], dXYs[:, k, 1],
                   angles='xy', scale_units='xy', scale=scale, alpha=alpha)
        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
        plt.title('K={} '.format(k) + mouse)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def get_z_percentage_by_grid(masks_a, z, K, G):
    masks_z_a = np.array([(z[:-1] + 1) * get_np(masks_a[g]) for g in range(G)])

    # (G, K) For each grid g, number of data in that grid = k
    grid_z_a = np.array([[sum(masks_z_a[g] == k) for k in range(1, K + 1)] for g in range(G)])

    grid_z_a_percentage = grid_z_a / (grid_z_a.sum(axis=1)[:, None] + 1e-6)

    return grid_z_a_percentage


def get_masks_for_two_animals(data, x_grids, y_grids):
    """
    :param data: (T, 4)
    :param x_grids
    :param y_grids
    :return: two lists of masks, each list contains G masks, where each mask is a binary-valued array of length T
    """

    data = check_and_convert_to_tensor(data)

    _, D = data.shape
    assert D == 4
    masks_a = get_masks_for_single_animal(data[:, 0:2], x_grids, y_grids)
    masks_b = get_masks_for_single_animal(data[:, 2:4], x_grids, y_grids)
    return masks_a, masks_b


def get_masks_for_single_animal(data_a, x_grids, y_grids):
    """
        :param data: (T, 2)
        :param x_grids
        :param y_grids
        :return: a lists which contains G masks, where each mask is a binary-valued array of length T
        """

    data = check_and_convert_to_tensor(data_a)

    masks_a = []
    for i in range(len(x_grids) - 1):
        for j in range(len(y_grids) - 1):
            if i == 0:
                cond_x = (x_grids[i] <= data[:, 0]) & (data[:, 0] <= x_grids[i + 1])
            else:
                cond_x = (x_grids[i] < data[:, 0]) & (data[:, 0] <= x_grids[i + 1])
            if j == 0:
                cond_y = (y_grids[j] <= data[:, 1]) & (data[:, 1] <= y_grids[j + 1])
            else:
                cond_y = (y_grids[j] < data[:, 1]) & (data[:, 1] <= y_grids[j + 1])
            mask = (cond_x & cond_y).double()
            masks_a.append(mask)

    masks_a = torch.stack(masks_a, dim=0)
    assert torch.all(masks_a.sum(dim=0) == 1)

    return masks_a


def find_Q_masks(angles):
    # mask the angles by quadrants
    # Put origin in the first quadrant
    assert angles.shape[1] == 2  # (T, 2)
    q1 = ((angles[:, 0] >= 0) & (angles[:, 1] >= 0)) # (T, )
    q2 = (angles[:, 0] < 0) & (angles[:, 1] > 0) # (T, )
    q3 = (angles[:, 0] < 0) & (angles[:, 1] <= 0) # (T, )
    q4 = (angles[:, 0] >= 0) & (angles[:, 1] < 0) # (T, )
    qs = np.stack((q1, q2, q3, q4), axis=0)  # (4, T)

    assert np.all(qs.sum(axis=0) == 1), "qs = {}, \n qs.sum(axis=0) = {}".format(qs, qs.sum(axis=0))
    return qs


def get_angles_single(data):
    """must make sure that data are consecutive"""
    assert data.shape[1] == 2  # (T, 2)
    start = data[:-1]
    end = data[1:]
    dXY = end - start  # (T-1, 2)
    angles = np.arctan(dXY[:, 1] / (dXY[:, 0] + 1e-8))  # (T-1, )

    angles_qs_add = [0, np.pi, np.pi, 2 * np.pi]

    qs = find_Q_masks(dXY)
    for i in range(4):
        angles[qs[i]] = angles[qs[i]] + angles_qs_add[i]
    return angles


def get_angles_single_from_quiver(dXY):
    angles = np.arctan(dXY[:, 1] / (dXY[:, 0] + 1e-8))  # (T-1, )

    angles_qs_add = [0, np.pi, np.pi, 2 * np.pi]

    qs = find_Q_masks(dXY)
    for i in range(4):
        angles[qs[i]] = angles[qs[i]] + angles_qs_add[i]
    return angles


def get_all_angles(data, x_grids, y_grids, device=torch.device('cpu')):

    dXY = data[1:] - data[:-1]
    if isinstance(dXY, torch.Tensor):
        dXY = get_np(dXY)

    return get_all_angles_from_quiver(data[:-1], dXY, x_grids, y_grids, device=device)


def get_all_angles_from_quiver(XY, dXY, x_grids, y_grids, device=torch.device('cpu')):
    # XY and dXY should have the same shape
    if isinstance(XY, np.ndarray):
        XY = torch.tensor(XY, dtype=torch.float64, device=device)

    _, D = XY.shape

    if D == 4:
        masks_a, masks_b = get_masks_for_two_animals(XY, x_grids, y_grids)
        masks_a = get_np(masks_a)
        masks_b = get_np(masks_b)

        angles_a = []
        angles_b = []
        G = (len(x_grids) - 1) * (len(y_grids) - 1)
        for g in range(G):
            dXY_a_g = dXY[masks_a[g] == 1][:, 0:2]
            dXY_b_g = dXY[masks_b[g] == 1][:, 2:4]

            angles_a.append(get_angles_single_from_quiver(dXY_a_g))
            angles_b.append(get_angles_single_from_quiver(dXY_b_g))

        return angles_a, angles_b
    elif D == 2:
        masks_a = get_masks_for_single_animal(XY, x_grids, y_grids)
        masks_a = get_np(masks_a)

        angles_a = []
        G = (len(x_grids) - 1) * (len(y_grids) - 1)
        for g in range(G):
            dXY_a_g = dXY[masks_a[g] == 1][:, 0:2]
            angles_a.append(get_angles_single_from_quiver(dXY_a_g))

        return angles_a
    else:
        raise ValueError("Invalid data shape")


def plot_angles(angles, title_name, n_x, n_y, bins=50, hist=True):
    plt.figure(figsize=(16, 12))
    plt.suptitle(title_name, fontsize=20)

    for i in range(n_x):
        for j in range(n_y):
            plot_idx = (n_y - j - 1) * n_x + i + 1
            plt.subplot(n_y, n_x, plot_idx)

            grid_idx = i * n_y + j
            if hist:
                plt.hist(angles[grid_idx], bins=bins)
            else:
                sns.kdeplot(angles[grid_idx])
            plt.xlim(0, 2 * np.pi)
            plt.xlim(0, 2 * np.pi)
            plt.xticks([0, np.pi / 2, np.pi, np.pi / 2 * 3, 2 * np.pi],
                       [0, r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])


def plot_list_of_angles(list_of_angles, labels, title_name, n_x, n_y):
    plt.figure(figsize=(16, 12))
    plt.suptitle(title_name, fontsize=20)

    colors = ['C0', 'C1', 'C2']
    custom_lines = [Line2D([0], [0], color='C0', lw=1),
                    Line2D([0], [0], color='C1', lw=1),
                    Line2D([0], [0], color='C2', lw=1)]

    for i in range(n_x):
        for j in range(n_y):
            plot_idx = (n_y - j - 1) * n_x + i + 1
            plt.subplot(n_y, n_x, plot_idx)

            grid_idx = i * n_y + j

            for angles, label, color in zip(list_of_angles, labels, colors):
                if angles[grid_idx].shape[0] > 10:
                    sns.kdeplot(angles[grid_idx], label=label, color=color, legend=False)
            if grid_idx == 0:
                plt.legend(custom_lines, ['data', 'sample', 'sample_c'])

            plt.xlim(0, 2 * np.pi)
            plt.xlim(0, 2 * np.pi)
            plt.xticks([0, np.pi / 2, np.pi, np.pi / 2 * 3, 2 * np.pi],
                       [0, r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])


def get_speed(data, x_grids, y_grids, device=torch.device('cpu')):
    # data (T, 4)
    _, D = data.shape
    assert D == 4 or D == 2
    if D == 4:
        speed_a = get_speed_for_single_animal(data[:, 0:2], x_grids, y_grids, device=device)
        speed_b = get_speed_for_single_animal(data[:, 2:4], x_grids, y_grids, device=device)
        return speed_a, speed_b
    else:
        speed_a = get_speed_for_single_animal(data, x_grids, y_grids)
        return speed_a


def get_speed_for_single_animal(data, x_grids, y_grids, device=torch.device('cpu')):
    # data: (T, 2)
    _, D = data.shape
    assert D == 2

    if isinstance(data, np.ndarray):
        diff = np.diff(data, axis=0)  # (T-1, 2)
        data = torch.tensor(data, dtype=torch.float64, device=device)
        masks_a = get_masks_for_single_animal(data[:-1], x_grids, y_grids)
    elif isinstance(data, torch.Tensor):
        masks_a = get_masks_for_single_animal(data[:-1], x_grids, y_grids)
        diff = np.diff(get_np(data), axis=0)  # (T-1, 2)
    else:
        raise ValueError("Data must be either np.ndarray or torch.Tensor")

    speed_a_all = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)  # (T-1, 2)

    speed_a = []
    G = (len(x_grids) - 1) * (len(y_grids) - 1)
    for g in range(G):
        speed_a_g = speed_a_all[get_np(masks_a[g]) == 1]

        speed_a.append(speed_a_g)

    return speed_a


def plot_list_of_speed(list_of_speed,  labels, title_name, n_x, n_y):
    plt.figure(figsize=(16, 12))
    plt.suptitle(title_name, fontsize=20)

    colors = ['C0', 'C1', 'C2']
    custom_lines = [Line2D([0], [0], color='C0', lw=1),
                    Line2D([0], [0], color='C1', lw=1),
                    Line2D([0], [0], color='C2', lw=1)]
    for i in range(n_x):
        for j in range(n_y):
            plot_idx = (n_y - j - 1) * n_x + i + 1
            plt.subplot(n_y, n_x, plot_idx)

            grid_idx = i * n_y + j

            for speed, label, color in zip(list_of_speed, labels, colors):
                if speed[grid_idx].shape[0] > 10:
                    sns.kdeplot(speed[grid_idx], label=label, color=color, legend=False)

            if grid_idx == 0:
                plt.legend(custom_lines, ['data', 'sample_x', 'sample_x_center'])


def test_plot_grid_and_weight_idx(n_x, n_y):

    plt.figure(figsize=(n_x * 5, n_y * 4))

    for j in range(n_y):
        for i in range(n_x):
            plt.subplot(n_y, n_x, (n_y - j - 1) * n_x + i + 1)

            plot_idx = (n_y - j - 1) * n_x + i + 1
            grid_idx = i * n_y + j
            plt.text(0.5, 0.5, "plot_index: {}".format(plot_idx), fontsize=12, color='k')

            plt.text(0.8, 0.3, "grid_index: {}".format(grid_idx), fontsize=12, color='k')
            plt.xticks(np.arange(0, 4, 1), ["lower L", "upper L", "lower R", "upper R"])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_space_dist(data, x_grids, y_grids, grid_alpha=1):
    # TODO: there are some
    T, D = data.shape
    assert D == 2 or D == 4
    if isinstance(data, torch.Tensor):
        data = get_np(data)

    n_levels = int(T / 36)

    if D == 4:
        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1)
        sns.kdeplot(data[:, 0], data[:, 1], n_levels=n_levels)
        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
        plt.title("virgin")

        plt.subplot(1, 2, 2)
        sns.kdeplot(data[:, 2], data[:, 3], n_levels=n_levels)
        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
        plt.title("mother")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        plt.figure(figsize=(8,7))

        sns.kdeplot(data[:,0], data[:,1], n_levels=n_levels)
        add_grid(x_grids, y_grids)

