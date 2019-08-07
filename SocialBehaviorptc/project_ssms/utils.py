import torch
import numpy as np

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
            x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True,
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
            x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True,
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
            x_predict = model.observation.sample_x(model_z[t], data[:t], return_np=True,
                                                   momentum_vec=momentum_vecs[t - 1],
                                                   interaction_vec=interaction_vecs[t - 1])
            x_predict_arr.append(x_predict)

        x_predict_arr = np.array(x_predict_arr)
        return x_predict_arr


def k_step_prediction_for_direction_model(model, model_z, data, momentum_vecs=None, features=None):
    return k_step_prediction_for_momentum_feature_model(model, model_z, data, momentum_vecs, features)
