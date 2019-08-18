import numpy as np

import matplotlib.pyplot as plt


def add_grid(x_grids, y_grids):
    if x_grids is None or y_grids is None:
        return
    plt.scatter([x_grids[0], x_grids[0], x_grids[-1], x_grids[-1]], [-10, 390, -10, 390])
    for j in range(len(y_grids)):
        plt.plot([x_grids[0], x_grids[-1]], [y_grids[j], y_grids[j]], '--', color='grey')

    for i in range(len(x_grids)):
        plt.plot([x_grids[i], x_grids[i]], [y_grids[0], y_grids[-1]], '--', color='grey')


def add_grid_to_ax(ax, x_grids, y_grids):
    ax.scatter([x_grids[0], x_grids[0], x_grids[-1], x_grids[-1]], [y_grids[0], y_grids[-1], y_grids[0], y_grids[-1]])
    for j in range(len(y_grids)):
        ax.plot([x_grids[0], x_grids[-1]], [y_grids[j], y_grids[j]], '--', color='grey')

    for i in range(len(x_grids)):
        ax.plot([x_grids[i], x_grids[i]], [y_grids[0], y_grids[-1]], '--', color='grey')


def plot_realdata_quiver(realdata, x_grids=None, y_grids=None, scale=0.3, alpha=0.8, xlim=None, ylim=None):
    assert isinstance(realdata, np.ndarray), "please input ndarray"
    start = realdata[:-1]
    end = realdata[1:]
    dXY = end - start

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.quiver(start[:, 0], start[:, 1], dXY[:, 0], dXY[:, 1],
               angles='xy', scale_units='xy', scale=scale, alpha=alpha)
    add_grid(x_grids, y_grids)
    plt.title("virgin")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.subplot(1, 2, 2)
    plt.quiver(start[:, 2], start[:, 3], dXY[:, 0], dXY[:, 1],
               angles='xy', scale_units='xy', scale=scale, alpha=alpha)
    add_grid(x_grids, y_grids)
    plt.title("mother")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)


def plot_weights(weights, Df, K, x_grids, y_grids, max_weight=10):
    assert Df == 4

    n_grid_x = len(x_grids) - 1
    n_grid_y = len(y_grids) - 1

    assert weights.shape == (n_grid_x*n_grid_y, K, Df)

    plt.figure(figsize=(n_grid_x*5, n_grid_y*4))

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


def plot_dynamics(weighted_corner_vecs, animal, x_grids, y_grids, K, scale=0.1, percentage=None):
    result_corner_vecs = np.sum(weighted_corner_vecs, axis=2)
    n_x = len(x_grids) - 1
    n_y = len(y_grids) - 1
    grid_centers = np.array(
        [[1 / 2 * (x_grids[i] + x_grids[i + 1]), 1 / 2 * (y_grids[j] + y_grids[j + 1])] for i in range(n_x) for j in
         range(n_y)])
    Df = 4

    def plot_dynamics_k(k):
        add_grid(x_grids, y_grids)

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
        for k in range(K):
            plt.subplot(1, K, k+1)
            plot_dynamics_k(k)
    elif 4 < K <= 8:
        plt.figure(figsize=(20, 10))
        for k in range(K):
            plt.subplot(2, int(K/2)+1, k+1)
            plot_dynamics_k(k)

    plt.tight_layout()


def plot_quiver(XYs, dXYs, mouse, K,scale=1, alpha=1):

    if K <= 5:
        plt.figure(figsize=(20, 4))

        for k in range(K):
            plt.subplot(1, K, k+1)
            plt.quiver(XYs[:, 0], XYs[:, 1], dXYs[:, k, 0], dXYs[:, k, 1],
                      angles='xy', scale_units='xy', scale=scale, alpha=alpha)
            plt.title('K={} '.format(k) + mouse)

    elif 5 < K <= 8:
        plt.figure(figsize=(20, 8))

        for k in range(K):
            plt.subplot(2, int(K/2)+1, k+1)
            plt.quiver(XYs[:, 0], XYs[:, 1], dXYs[:, k, 0], dXYs[:, k, 1],
                       angles='xy', scale_units='xy', scale=scale, alpha=alpha)
            plt.title('K={} '.format(k) + mouse)

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