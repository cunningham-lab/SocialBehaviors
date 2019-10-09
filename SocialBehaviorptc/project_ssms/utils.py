import torch
import numpy as np
import datetime

from ssm_ptc.utils import k_step_prediction, check_and_convert_to_tensor


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
    data = check_and_convert_to_tensor(data)

    if feature_vecs is None or gridpoints_idx is None or gridpoints is None:
        print("Did not provide memory information")
        return k_step_prediction(model, model_z, data)
    else:
        grid_points_idx_a, grid_points_idx_b = gridpoints_idx
        gridpoints_a, gridpoints_b = gridpoints
        feature_vecs_a, feature_vecs_b = feature_vecs

        x_predict_arr = []
        x_predict = model.observation.sample_x(model_z[0], data[:0], return_np=True)
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


def downsample(traj, n):
    # data : (T, D)
    if n == 1:
        return traj
    T = len(traj)
    T_over_n = int(T / n)
    idx = n * np.arange(T_over_n)
    return traj[idx]

