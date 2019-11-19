import torch
import numpy as np
import datetime

from ssm_ptc.utils import k_step_prediction, check_and_convert_to_tensor, get_np


def k_step_prediction_for_momentum_feature_model(model, model_z, data, momentum_vecs=None, features=None):
    data = check_and_convert_to_tensor(data)

    if momentum_vecs is None or features is None:
        return k_step_prediction(model, model_z, data)
    else:
        x_predict_arr = []
        x_predict = model.observation.sample_x(model_z[0], data[:0], return_np=True)
        x_predict_arr.append(x_predict)
        for t in range(1, data.shape[0]):
            x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True, transformation=True,
                                                   momentum_vec=momentum_vecs[t-1],
                                                   features=(features[0][t-1], features[1][t-1]))
            x_predict_arr.append(x_predict)

        x_predict_arr = np.array(x_predict_arr)
        return x_predict_arr


def k_step_prediction_for_momentum_model(model, model_z, data, momentum_vecs=None):
    data = check_and_convert_to_tensor(data)

    if momentum_vecs is None:
        return k_step_prediction(model, model_z, data)
    else:
        x_predict_arr = []
        x_predict = model.observation.sample_x(model_z[0], data[:0], return_np=True)
        x_predict_arr.append(x_predict)
        for t in range(1, data.shape[0]):
            x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True, transformation=True,
                                                   momentum_vec=momentum_vecs[t-1])
            x_predict_arr.append(x_predict)

        x_predict_arr = np.array(x_predict_arr)
        return x_predict_arr


def k_step_prediction_for_momentum_interaction_model(model, model_z, data, momentum_vecs=None,
                                                     interaction_vecs=None):
    data = check_and_convert_to_tensor(data)

    if momentum_vecs is None or interaction_vecs is None:
        return k_step_prediction(model, model_z, data)
    else:
        x_predict_arr = []
        x_predict = model.observation.sample_x(model_z[0], data[:0], return_np=True)
        x_predict_arr.append(x_predict)
        for t in range(1, data.shape[0]):
            x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True, transformation=True,
                                                   momentum_vec=momentum_vecs[t - 1],
                                                   interaction_vec=interaction_vecs[t - 1])
            x_predict_arr.append(x_predict)

        x_predict_arr = np.array(x_predict_arr)
        return x_predict_arr


def k_step_prediction_for_direction_model(model, model_z, data, momentum_vecs=None, features=None):
    return k_step_prediction_for_momentum_feature_model(model, model_z, data, momentum_vecs, features)


def k_step_prediction_for_grid_model(model, model_z, data, **memory_kwargs):
    if len(data) == 0:
        return None
    data = check_and_convert_to_tensor(data)

    memory_kwargs_a = memory_kwargs.get("memory_kwargs_a", None)
    memory_kwargs_b = memory_kwargs.get("memory_kwargs_b", None)
    if memory_kwargs_a is None or memory_kwargs_b is None:
        print("Did not provide memory information")
        return k_step_prediction(model, model_z, data)
    else:
        momentum_vecs_a = memory_kwargs_a.get("momentum_vecs", None)
        feature_vecs_a = memory_kwargs_a.get("feature_vecs", None)

        momentum_vecs_b = memory_kwargs_b.get("momentum_vecs", None)
        feature_vecs_b = memory_kwargs_b.get("feature_vecs", None)

        x_predict_arr = []
        x_predict = model.observation.sample_x(model_z[0], data[:0], return_np=True)
        x_predict_arr.append(x_predict)
        for t in range(1, data.shape[0]):
            if momentum_vecs_a is None:
                m_kwargs_a = dict(feature_vec=feature_vecs_a[t - 1])
                m_kwargs_b = dict(feature_vec=feature_vecs_b[t - 1])
            else:
                m_kwargs_a = dict(momentum_vec=momentum_vecs_a[t - 1], feature_vec=feature_vecs_a[t - 1])
                m_kwargs_b = dict(momentum_vec=momentum_vecs_b[t - 1], feature_vec=feature_vecs_b[t - 1])

            x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True, transformation=True,
                                                   memory_kwargs_a=m_kwargs_a, memory_kwargs_b=m_kwargs_b)
            x_predict_arr.append(x_predict)

        x_predict_arr = np.array(x_predict_arr)
        return x_predict_arr


def k_step_prediction_for_lineargrid_model(model, model_z, data, gridpoints=None, gridpoints_idx=None, feature_vecs=None):
    if len(data) == 0:
        return None
    data = check_and_convert_to_tensor(data)

    if feature_vecs is None or gridpoints_idx is None or gridpoints is None:
        print("Did not provide memory information")
        return k_step_prediction(model, model_z, data)
    else:
        grid_points_idx_a, grid_points_idx_b = gridpoints_idx
        gridpoints_a, gridpoints_b = gridpoints
        feature_vecs_a, feature_vecs_b = feature_vecs

        x_predict_arr = []
        x_predict = model.observation.sample_x(model_z[0], data[:0], return_np=True, transformation=True)
        x_predict_arr.append(x_predict)
        for t in range(1, data.shape[0]):
            grid_points_idx_t = (grid_points_idx_a[t - 1], grid_points_idx_b[t - 1])
            gridpoints_t = (gridpoints_a[t - 1], gridpoints_b[t - 1])
            feature_vec_t = (feature_vecs_a[t - 1:t], feature_vecs_b[t - 1:t])

            x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True, transformation=True,
                                                   gridpoints=gridpoints_t,
                                                 gridpoints_idx=grid_points_idx_t, feature_vec=feature_vec_t)
            x_predict_arr.append(x_predict)

        x_predict_arr = np.array(x_predict_arr)
        return x_predict_arr


def k_step_prediction_for_weightedgrid_model(model, model_z, data, distances_a=None, distances_b=None, feature_vecs=None):
    if len(data) == 0:
        return None
    data = check_and_convert_to_tensor(data)

    if feature_vecs is None or distances_a is None or distances_b is None:
        print("Did not provide memory information")
        return k_step_prediction(model, model_z, data)
    else:
        feature_vecs_a, feature_vecs_b = feature_vecs

        x_predict_arr = []
        x_predict = model.observation.sample_x(model_z[0], data[:0], return_np=True)
        x_predict_arr.append(x_predict)
        for t in range(1, data.shape[0]):
            feature_vec_t = (feature_vecs_a[t - 1:t], feature_vecs_b[t - 1:t])

            x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True, transformation=True,
                                                   distance_a=distances_a[t-1:t], distance_b=distances_b[t-1:t],
                                                   feature_vec=feature_vec_t)
            x_predict_arr.append(x_predict)

        x_predict_arr = np.array(x_predict_arr)
        return x_predict_arr

def k_step_prediction_for_gpgrid_model(model, model_z, data, **memory_kwargs):
    data = check_and_convert_to_tensor(data)

    if memory_kwargs == {}:
        print("Did not provide memory information")
        return k_step_prediction(model, model_z, data)
    else:

        feature_vecs_a =memory_kwargs.get("feature_vecs_a", None)
        feature_vecs_b = memory_kwargs.get("feature_vecs_b", None)
        gpt_idx_a = memory_kwargs.get("gpt_idx_a", None)
        gpt_idx_b= memory_kwargs.get("gpt_idx_b", None)
        grid_idx_a = memory_kwargs.get("grid_idx_a", None)
        grid_idx_b = memory_kwargs.get("grid_idx_b")
        coeff_a = memory_kwargs.get("coeff_a", None)
        coeff_b = memory_kwargs.get("coeff_b", None)
        dist_sq_a = memory_kwargs.get("dist_sq_a", None)
        dist_sq_b = memory_kwargs.get("dist_sq_b", None)



        x_predict_arr = []
        x_predict = model.observation.sample_x(model_z[0], data[:0], return_np=True)
        x_predict_arr.append(x_predict)
        for t in range(1, data.shape[0]):
            if dist_sq_a is None:
                x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True, transformation=True,
                                                       feature_vec_a=feature_vecs_a[t-1:t], feature_vec_b=feature_vecs_b[t-1:t],
                                                       gpt_idx_a=gpt_idx_a[t-1:t], gpt_idx_b=gpt_idx_b[t-1:t],
                                                       grid_idx_a=grid_idx_a[t-1:t], grid_idx_b=grid_idx_b[t-1:t],
                                                       coeff_a=coeff_a[t-1:t], coeff_b=coeff_b[t-1:t])
            else:
                x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True, transformation=True,
                                                       feature_vec_a=feature_vecs_a[t - 1:t],
                                                       feature_vec_b=feature_vecs_b[t - 1:t],
                                                       gpt_idx_a=gpt_idx_a[t - 1:t], gpt_idx_b=gpt_idx_b[t - 1:t],
                                                       grid_idx_a=grid_idx_a[t - 1:t], grid_idx_b=grid_idx_b[t - 1:t],
                                                       dist_sq_a=dist_sq_a[t - 1:t], dist_sq_b=dist_sq_b[t - 1:t])
            x_predict_arr.append(x_predict)

        x_predict_arr = np.array(x_predict_arr)
        return x_predict_arr


def k_step_prediction_for_gpmodel(model, model_z, data, **memory_kwargs):
    data = check_and_convert_to_tensor(data)

    T, D = data.shape
    assert D == 4 or D == 2, D

    K = model.observation.K

    if memory_kwargs == {}:
        print("Did not provide memory information")
        return k_step_prediction(model, model_z, data)
    else:

        # compute As
        if D == 4:
            _, A_a = model.observation.get_gp_cache(data[:-1, 0:2], 0, A_only=True, **memory_kwargs)
            _, A_b = model.observation.get_gp_cache(data[:-1, 2:4], 1, A_only=True, **memory_kwargs)
            assert A_a.shape == A_b.shape == (T-1, K, 2, model.observation.n_gps*2), "{}, {}".format(A_a.shape, A_b.shape)

            x_predict_arr = []
            x_predict = model.observation.sample_x(model_z[0], data[:0], return_np=True)
            x_predict_arr.append(x_predict)
            for t in range(1, data.shape[0]):
                x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True, transformation=True,
                                                       A_a=A_a[t-1:t, model_z[t]], A_b=A_b[t-1:t, model_z[t]])
                x_predict_arr.append(x_predict)
        else:
            _, A = model.observation.get_gp_cache(data[:-1], A_only=True, **memory_kwargs)
            assert A.shape == (T - 1, K, 2, model.observation.n_gps * 2), A.shape

            x_predict_arr = []
            x_predict = model.observation.sample_x(model_z[0], data[:0], return_np=True)
            x_predict_arr.append(x_predict)
            for t in range(1, data.shape[0]):
                x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True, transformation=True,
                                                       A=A[t - 1:t, model_z[t]])
                x_predict_arr.append(x_predict)

        x_predict_arr = np.array(x_predict_arr)
        return x_predict_arr


def k_step_prediction_for_lstm_model(model, model_z, data, feature_vecs=None):
    data = check_and_convert_to_tensor(data)

    if feature_vecs is None:
        print("Did not provide memory information")
        return k_step_prediction(model, model_z, data)
    else:
        feature_vecs_a, feature_vecs_b = feature_vecs

        x_predict_arr = []
        x_predict = model.observation.sample_x(model_z[0], data[:0], return_np=True)
        x_predict_arr.append(x_predict)
        for t in range(1, data.shape[0]):
            feature_vec_t = (feature_vecs_a[t - 1:t], feature_vecs_b[t - 1:t])

            x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True, transformation=True,
                                                   feature_vec=feature_vec_t)
            x_predict_arr.append(x_predict)

        x_predict_arr = np.array(x_predict_arr)
        return x_predict_arr


def k_step_prediction_for_lstm_based_model(model, model_z, data, k=0, feature_vecs=None):
    data = check_and_convert_to_tensor(data)
    T, D = data.shape

    lstm_states = {}

    x_predict_arr = []
    if k == 0:
        if feature_vecs is None:
            print("Did not provide memory information")
            return k_step_prediction(model, model_z, data)
        else:
            feature_vecs_a, feature_vecs_b = feature_vecs

            x_predict = model.observation.sample_x(model_z[0], data[:0], return_np=True)
            x_predict_arr.append(x_predict)
            for t in range(1, data.shape[0]):
                feature_vec_t = (feature_vecs_a[t - 1:t], feature_vecs_b[t - 1:t])

                x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True, transformation=True,
                                                       feature_vec=feature_vec_t, lstm_states=lstm_states)
                x_predict_arr.append(x_predict)
    else:
        assert k > 0
        # neglects t = 0 since there is no history

        if T <= k:
            raise ValueError("Please input k such that k < {}.".format(T))

        for t in range(1, T - k + 1):
            # sample k steps forward
            # first step use real value
            z, x = model.sample(1, prefix=(model_z[t-1:t], data[t-1:t]), return_np=False, transformation=True,
                                      lstm_states=lstm_states)
            # last k-1 steps use sampled value
            if k>=1:
                sampled_lstm_states = dict(h_t=lstm_states["h_t"], c_t=lstm_states["c_t"])
                for i in range(k-1):
                    z, x = model.sample(1, prefix=(z, x), return_np=False, transformation=True,
                                        lstm_states=sampled_lstm_states)
            assert x.shape == (1, D)
            x_predict_arr.append(get_np(x[0]))

    x_predict_arr = np.array(x_predict_arr)
    assert x_predict_arr.shape == (T-k, D)
    return x_predict_arr


def downsample(traj, n):
    # data : (T, D)
    if n == 1:
        return traj
    T = len(traj)
    T_over_n = int(T / n)
    idx = n * np.arange(T_over_n)
    return traj[idx]


def clip(val, boundary, eps=0.1, device=torch.device('cpu')):
    if val <= boundary[0]:
        val = boundary[0] + eps * torch.rand(1, dtype=torch.float64, device=device)
    elif val >= boundary[1]:
        val = boundary[1] - eps * torch.rand(1, dtype=torch.float64, device=device)
    return val
