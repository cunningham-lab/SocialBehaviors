import torch
import numpy as np


def fit_line(xs, ys, return_b=False):
    """
    :param xs: (momentum_lags, )
    :param ys: (momentum_lags, )
    :param return_b: whether or not to return the intercept
    :return:
    """

    "least square with verticle offsetds"
    "y = ax + b, slope -- a, y-intercept -- b"
    xbar = torch.mean(xs, dim=-1)  # ()
    ybar = torch.mean(ys, dim=-1)  # ()
    a = torch.sum((xs - xbar) * (ys - ybar)) / torch.sum((xs - xbar) ** 2)  # (momentum_lags, )
    if not return_b:
        return a
    b = ybar - a * xbar
    return a, b


def fit_line_in_batch(batch_xs, batch_ys, return_b=False):
    """

    :param batch_xs: (batch_size, momentum_lags)
    :param batch_ys: (batch_size, momentum_lags)
    :param return_b: whether or not to return the intercept
    :return:
    """

    "least square with verticle offsetds"
    "y = ax + b, slope -- a, y-intercept -- b"

    batch_size = batch_xs.shape[0]

    xbar = torch.mean(batch_xs, dim=-1, keepdim=True)  # (batch_size, 1)
    ybar = torch.mean(batch_ys, dim=-1, keepdim=True)  # (batch_size, 1)
    a = torch.sum((batch_xs - xbar) * (batch_ys - ybar), dim=-1) / torch.sum((batch_xs - xbar) ** 2, dim=-1)  # (batch_size, )
    assert a.shape == (batch_size, )

    if not return_b:
        return a
    b = ybar - a * xbar  # (batch_size, )
    return a, b


def weighted_fit_line(xs, ys, weights=None, return_b=False):
    """
        :param xs: (momentum_lags, )
        :param ys: (momentum_lags, )
        :param return_b: whether or not to return the intercept
        :return:
        """
    "weighted least square"
    lags = xs.shape[0]
    if weights is None:
        weights = torch.ones(lags, dtype=torch.float64)
    else:
        assert weights.shape == (lags, )

    wxy = weights * xs * ys
    wx = weights * xs
    wy = weights * ys
    numerator = torch.sum(wxy) - torch.sum(wx) * torch.sum(wy) / torch.sum(weights)
    denominator = torch.sum(weights * xs * xs) - torch.sum(weights * xs)**2 / torch.sum(weights)
    a = numerator / denominator
    if not return_b:
        return a
    b = torch.sum(weights * (ys - a * xs)) / torch.sum(weights)
    return a, b


def weighted_fit_line_in_batch(batch_xs, batch_ys, weights=None, return_b=False):
    """
    :param batch_xs: (batch_size, momentum_lags)
    :param batch_ys: (batch_size, momentum_lags)
    :param return_b: whether or not to return the intercept
    :return:
    """

    "weighted least square"

    batch_size = batch_xs.shape[0]
    lags = batch_xs.shape[1]

    if weights is None:
        weights = torch.ones(lags, dtype=torch.float64)
    else:
        assert weights.shape == (lags, )

    wxy = weights * batch_xs * batch_ys  # (batch_size, momentum_lags)
    wx = weights * batch_xs  # (batch_size, momentum_lags)
    wy = weights * batch_ys  # (batch_size, momentum_lags)

    numerator = torch.sum(wxy, dim=-1) - torch.sum(wx, dim=-1) * torch.sum(wy, dim=-1) / torch.sum(weights) # (batch_size, )
    denominator = torch.sum(weights * batch_xs * batch_xs, dim=-1) - torch.sum(weights * batch_xs, dim=-1) ** 2 / torch.sum(weights)

    a = numerator / denominator  # (batch_size, )

    assert a.shape == (batch_size,)

    if not return_b:
        return a
    b = torch.sum(weights * (batch_ys - a * batch_xs), dim=-1) / torch.sum(weights)  # (batch_size, )
    return a, b


def filter_speed(traj, quantile=0.99, threshold=None):
    assert len(traj.shape) == 2
    assert traj.shape[1] == 2
    speed = np.sqrt(np.diff(traj[:, 0]) ** 2 + np.diff(traj[:, 1]) ** 2)
    if threshold is None:
        threshold = np.quantile(speed, quantile)

    part_okidx = np.where(speed < threshold)[0]
    part_okindex = np.zeros((part_okidx.shape[0] + 1,), dtype=part_okidx.dtype)
    part_okindex[1:] = part_okidx + 1

    time_index = np.arange(0, 36000)
    part_oktraj = traj[part_okindex]

    filtered_x = np.interp(time_index, part_okindex, part_oktraj[:, 0])
    filtered_y = np.interp(time_index, part_okindex, part_oktraj[:, 1])
    filtered_part_traj = np.concatenate((filtered_x[:, np.newaxis], filtered_y[:, np.newaxis]), axis=1)

    return filtered_part_traj


def filter_traj_by_speed(traj, q1, q2, t1=None, t2=None):
    filtered_traj_a = filter_speed(traj[:, 0:2], q1, t1)
    filtered_traj_b = filter_speed(traj[:, 2:4], q2, t2)
    filtered_traj = np.concatenate((filtered_traj_a, filtered_traj_b), axis=1)
    return filtered_traj


def get_momentum_in_batch(data, lags, weights):
    """Compute normalized momentum vector for 2D trajectories,
     where "normalized" means "normalized by momentum_lags". """

    assert data.shape[1] == 2
    assert weights.shape == (lags, )

    T = data.shape[0]

    vecs_init = torch.tensor([[0.0, 0.0]], dtype=torch.float64)

    if T == 1:
        # no momentum to accumulate
        return vecs_init

    if T < lags:
        vecs = [vecs_init[0]]
        for t in range(2, T+1):
            # actual momentum_lags = t
            xs, ys = data[0:t, 0], data[0:t, 1]  # (t, )
            slope = weighted_fit_line(xs, ys, weights[-t:])
            vec = torch.tensor([(xs[-1] - xs[0]) / t, slope * (xs[-1] - xs[0]) / t])
            vecs.append(vec)

        vecs = torch.stack(vecs, dim=0)

    else:
        # [2, momentum_lags)
        vecs_1 = []

        for t in range(2, lags):
            # actual momentum_lags = t
            xs, ys = data[0:t, 0], data[0:t, 1]  # (t, )
            slope = weighted_fit_line(xs, ys, weights[-t:])

            vec = torch.tensor([(xs[-1] - xs[0]) / t, slope * (xs[-1] - xs[0]) / t])

            vecs_1.append(vec)

        vecs_1 = torch.stack(vecs_1, dim=0)  #  (batch_size, 2)

        # [momentum_lags, T+1): can processed in batch
        batch_xs = torch.stack([data[t-lags:t, 0] for t in range(lags, T+1)])
        batch_ys = torch.stack([data[t-lags:t, 1] for t in range(lags, T+1)])

        slopes = weighted_fit_line_in_batch(batch_xs, batch_ys, weights, return_b=False)  # (batch_size, )

        starts = (batch_xs[..., -1] - batch_xs[..., 0]) / lags  # (batch_size, )
        ends = slopes * starts  # (batch_size, )

        vecs_2 = torch.stack((starts, ends), dim=-1)  # (batch_size, 2)

        vecs = torch.cat((vecs_init, vecs_1, vecs_2), dim=0)

    assert vecs.shape == (T, 2)

    return vecs


def get_momentum(data, lags, weights):
    """Compute a single normalized momentum vector based on the past trajectories """

    assert data.shape[1] == 2
    assert weights.shape == (lags, )

    T = data.shape[0]

    if T == 1:
        # no momentum to accumulate
        return torch.tensor([0.0, 0.0], dtype=torch.float64)

    if T < lags:
        xs, ys = data[:, 0], data[:, 1]
        slope = weighted_fit_line(xs, ys, weights[-T:])
        vec = torch.tensor([(xs[-1] - xs[0]) / T, slope * (xs[-1] - xs[0]) / T])

    else:
        xs, ys = data[T-lags:, 0], data[T-lags:, 1]
        slope = weighted_fit_line(xs, ys, weights)

        vec = torch.tensor([(xs[-1] - xs[0]) / lags, slope * (xs[-1] - xs[0]) / lags])

    assert vec.shape == (2, )

    return vec



