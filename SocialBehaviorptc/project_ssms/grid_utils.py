import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from ssm_ptc.utils import check_and_convert_to_tensor
from project_ssms.plot_utils import get_colors_and_cmap


def add_grid(x_grids, y_grids, grid_alpha=1):
    if x_grids is None or y_grids is None:
        return
    if isinstance(x_grids, torch.Tensor):
        x_grids = x_grids.numpy()
    if isinstance(y_grids, torch.Tensor):
        y_grids = y_grids.numpy()

    plt.scatter([x_grids[0], x_grids[0], x_grids[-1], x_grids[-1]], [-10, 390, -10, 390], alpha=grid_alpha)
    for j in range(len(y_grids)):
        plt.plot([x_grids[0], x_grids[-1]], [y_grids[j], y_grids[j]], '--', color='grey', alpha=grid_alpha)

    for i in range(len(x_grids)):
        plt.plot([x_grids[i], x_grids[i]], [y_grids[0], y_grids[-1]], '--', color='grey', alpha=grid_alpha)


def add_grid_to_ax(ax, x_grids, y_grids):
    if isinstance(x_grids, torch.Tensor):
        x_grids = x_grids.numpy()
    if isinstance(y_grids, torch.Tensor):
        y_grids = y_grids.numpy()

    ax.scatter([x_grids[0], x_grids[0], x_grids[-1], x_grids[-1]], [y_grids[0], y_grids[-1], y_grids[0], y_grids[-1]])
    for j in range(len(y_grids)):
        ax.plot([x_grids[0], x_grids[-1]], [y_grids[j], y_grids[j]], '--', color='grey')

    for i in range(len(x_grids)):
        ax.plot([x_grids[i], x_grids[i]], [y_grids[0], y_grids[-1]], '--', color='grey')


def plot_realdata_quiver(realdata, z, K, x_grids=None, y_grids=None,
                         xlim=None, ylim=None, title=None, grid_alpha=1, **quiver_args):
    if isinstance(realdata, torch.Tensor):
        realdata = realdata.numpy()

    start = realdata[:-1]
    end = realdata[1:]
    dXY = end - start

    h = 1 / K
    ticks = [(1 / 2 + k) * h for k in range(K)]
    colors, cm = get_colors_and_cmap(K)

    if realdata.shape[-1] == 2:
        plt.figure(figsize=(8, 7))
        if title is not None:
            plt.suptitle(title)

        plt.quiver(start[:, 0], start[:, 1], dXY[:, 0], dXY[:, 1],
                   angles='xy', scale_units='xy', cmap=cm, color=colors[z], **quiver_args)
        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        cb = plt.colorbar(label='k', ticks=ticks)
        cb.set_ticklabels(range(K))
    else:
        assert realdata.shape[-1] == 4

        plt.figure(figsize=(16, 7))
        if title is not None:
            plt.suptitle(title)

        plt.subplot(1, 2, 1)
        plt.quiver(start[:, 0], start[:, 1], dXY[:, 0], dXY[:, 1],
               angles='xy', scale_units='xy', cmap=cm, color=colors[z], **quiver_args)
        cb = plt.colorbar(label='k', ticks=ticks)
        cb.set_ticklabels(range(K))

        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
        plt.title("virgin")
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.subplot(1, 2, 2)
        plt.quiver(start[:, 2], start[:, 3], dXY[:, 2], dXY[:, 3],
                   angles='xy', scale_units='xy', cmap=cm, color=colors[z], **quiver_args)
        cb = plt.colorbar(label='k', ticks=ticks)
        cb.set_ticklabels(range(K))

        add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
        plt.title("mother")
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
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

    plt.tight_layout()


def add_percentage(k, percentage, grid_centers):
    if percentage is None:
        return

    G = len(percentage)

    texts = ["{0:.2f}%".format(percentage[g][k] * 100) for g in range(G)]

    for c, text in zip(grid_centers, texts):
        plt.text(c[0] + 20, c[1] + 40, text, fontsize=12, color='k')


def plot_dynamics(weighted_corner_vecs, animal, x_grids, y_grids, K, scale=0.1, percentage=None, title=None,
                  grid_alpha=1):
    if isinstance(x_grids, torch.Tensor):
        x_grids = x_grids.numpy()
    if isinstance(y_grids, torch.Tensor):
        y_grids = y_grids.numpy()

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

    plt.tight_layout()


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

    plt.tight_layout()


def get_z_percentage_by_grid(masks_a, z, K, G):
    masks_z_a = np.array([(z[:-1] + 1) * masks_a[g].numpy() for g in range(G)])

    # (G, K) For each grid g, number of data in that grid = k
    grid_z_a = np.array([[sum(masks_z_a[g] == k) for k in range(1, K + 1)] for g in range(G)])

    grid_z_a_percentage = grid_z_a / (grid_z_a.sum(axis=1)[:, None] + 1e-6)

    return grid_z_a_percentage


def get_masks(data, x_grids, y_grids):
    """
    :param data: (T, 4)
    :param x_grids
    :param y_grids
    :return: two lists of masks, each list contains G masks, where each mask is a binary-valued array of length T
    """

    data = check_and_convert_to_tensor(data)
    masks_a = []
    masks_b = []
    for i in range(len(x_grids)-1):
        for j in range(len(y_grids)-1):
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

            if i == 0:
                cond_x = (x_grids[i] <= data[:, 2]) & (data[:, 2] <= x_grids[i + 1])
            else:
                cond_x = (x_grids[i] < data[:, 2]) & (data[:, 2] <= x_grids[i + 1])
            if j == 0:
                cond_y = (y_grids[j] <= data[:, 3]) & (data[:, 3] <= y_grids[j + 1])
            else:
                cond_y = (y_grids[j] < data[:, 3]) & (data[:, 3] <= y_grids[j + 1])
            mask = (cond_x & cond_y).double()
            masks_b.append(mask)

    masks_a = torch.stack(masks_a, dim=0)
    assert torch.all(masks_a.sum(dim=0) == 1)
    masks_b = torch.stack(masks_b, dim=0)
    assert torch.all(masks_b.sum(dim=0) == 1)

    return masks_a, masks_b


def find_Q_masks(angles):
    # mask the angles by quadrants
    # Put origin in the first quadrant
    assert angles.shape[1] == 2  # (T, 2)
    q1 = ((angles[:, 0] >= 0) & (angles[:, 1] >= 0))
    q2 = (angles[:, 0] <= 0) & (angles[:, 1] > 0)
    q3 = (angles[:, 0] < 0) & (angles[:, 1] <= 0)
    q4 = (angles[:, 0] >= 0) & (angles[:, 1] < 0)
    qs = np.stack((q1, q2, q3, q4), axis=0)
    #print(qs)
    assert np.all(qs.sum(axis=0) == 1)
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


def get_all_angles(data, x_grids, y_grids):

    dXY = data[1:] - data[:-1]
    if isinstance(dXY, torch.Tensor):
        dXY = dXY.numpy()

    return get_all_angles_from_quiver(data[:-1], dXY, x_grids, y_grids)


def get_all_angles_from_quiver(XY, dXY, x_grids, y_grids):
    # XY and dXY should have the same shape
    if isinstance(XY, np.ndarray):
        XY = torch.tensor(XY, dtype=torch.float64)
    masks_a, masks_b = get_masks(XY, x_grids, y_grids)
    masks_a = masks_a.numpy()
    masks_b = masks_b.numpy()

    angles_a = []
    angles_b = []
    G = (len(x_grids) - 1) * (len(y_grids) - 1)
    for g in range(G):
        dXY_a_g = dXY[masks_a[g] == 1][:, 0:2]
        dXY_b_g = dXY[masks_b[g] == 1][:, 2:4]

        angles_a.append(get_angles_single_from_quiver(dXY_a_g))
        angles_b.append(get_angles_single_from_quiver(dXY_b_g))

    return angles_a, angles_b


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


def get_speed(data, x_grids, y_grids):
    # data (T, 4)
    if isinstance(data, np.ndarray):
        diff = np.diff(data, axis=0)  # (T-1, 4)
        data = torch.tensor(data, dtype=torch.float64)
        masks_a, masks_b = get_masks(data[:-1], x_grids, y_grids)
    elif isinstance(data, torch.Tensor):
        masks_a, masks_b = get_masks(data[:-1], x_grids, y_grids)
        diff = np.diff(data.numpy(), axis=0)  # (T-1, 4)
    else:
        raise ValueError("Data must be either np.ndarray or torch.Tensor")

    speed_a_all = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)  # (T-1, 2)
    speed_b_all = np.sqrt(diff[:, 2] ** 2 + diff[:, 3] ** 2)  # (T-1, 2)

    speed_a = []
    speed_b = []
    G = (len(x_grids) - 1) * (len(y_grids) - 1)
    for g in range(G):
        speed_a_g = speed_a_all[masks_a[g].numpy() == 1]
        speed_b_g = speed_b_all[masks_b[g].numpy() == 1]

        speed_a.append(speed_a_g)
        speed_b.append(speed_b_g)

    return speed_a, speed_b


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


def plot_space_dist(data, x_grids, y_grids, grid_alpha=1):
    # TODO: there are some
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    T = data.shape[0]
    n_levels = int(T / 36)
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    sns.kdeplot(data[:, 0], data[:, 1], n_levels=n_levels)
    add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
    plt.title("virgin")

    plt.subplot(1, 2, 2)
    sns.kdeplot(data[:, 2], data[:, 3], n_levels=n_levels)
    add_grid(x_grids, y_grids, grid_alpha=grid_alpha)
    plt.title("mother")

    plt.tight_layout()


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

    plt.tight_layout()


