import torch
import numpy as np

from ssm_ptc.utils import check_and_convert_to_tensor


def fit_line(xs, ys, return_b=False):
    """

    :param xs: (batch_size, lags)
    :param ys: (batch_size, lags)
    :param return_b: whether or not to return the intercept
    :return:
    """

    "least square with verticle offsetds"
    "y = ax + b, slope -- a, y-interpect -- b"
    xbar = torch.mean(xs, dim=-1)  # (batch_size, )
    ybar = torch.mean(ys, dim=-1)  # (batch_size, )
    a = torch.sum((xs - xbar) * (ys - ybar)) / torch.sum((xs - xbar) ** 2)
    b = ybar - a * xbar
    if not return_b:
        return a
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


def get_momentum(data, lags):
    """Compute normalized momentum vector for 2D trajectories, where "normalized" means "normalized by lags". """

    data = check_and_convert_to_tensor(data)

    assert data.shape[1] == 2

    T = data.shape[0]

    if T == 1:
        # no momentum to accumulate
        return np.array([0.0])

    normalized_momentum_vectors = [torch.tensor([0.0, 0.0], dtype=torch.float64)]
    if T < lags:
        for t in range(2, T):
            # actual lags = t
            xs, ys = data[0:t, 0], data[0:t, 1]
            slope = fit_line(xs, ys)

            vec = torch.tensor([(xs[-1] - xs[0]) / t,  slope * (xs[-1] - xs[0]) / t])

            normalized_momentum_vectors.append(vec)

    else:
        for t in range(2, lags):
            # actual lags = t
            xs, ys = data[0:t, 0], data[0:t, 1]
            slope = fit_line(xs, ys)

            vec = torch.tensor([(xs[-1] - xs[0]) / t, slope * (xs[-1] - xs[0]) / t])

            normalized_momentum_vectors.append(vec)

        for t in range(lags, T + 1):
            xs, ys = data[t - lags:t, 0], data[t - lags:t, 1]
            slope = fit_line(xs, ys)

            vec = torch.tensor([(xs[-1] - xs[0]) / lags,  slope * (xs[-1] - xs[0]) / lags])

            normalized_momentum_vectors.append(vec)

    assert len(normalized_momentum_vectors) == T

    out = torch.stack(normalized_momentum_vectors, dim=0)

    return out
