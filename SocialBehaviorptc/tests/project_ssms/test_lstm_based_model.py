import torch
import numpy as np

from project_ssms.coupled_transformations.lstm_based_transformation import LSTMBasedTransformation
from project_ssms.feature_funcs import f_corner_vec_func, feature_direction_vec
from project_ssms.ar_truncated_normal_observation import ARTruncatedNormalObservation
from project_ssms.utils import k_step_prediction_for_lstm_based_model

from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import k_step_prediction

import matplotlib.pyplot as plt


def test_tran():
    torch.manual_seed(0)
    np.random.seed(0)

    T = 100
    D = 4

    #data = np.array([[1.0, 1.0, 1.0, 6.0], [3.0, 6.0, 8.0, 6.0],
     #                [4.0, 7.0, 8.0, 5.0], [6.0, 7.0, 5.0, 6.0], [8.0, 2.0, 6.0, 1.0]])
    data = np.random.randn(T, D)
    data = torch.tensor(data, dtype=torch.float64)

    xmax = max(np.max(data[:,0].numpy()), np.max(data[:,2].numpy()))
    xmin = min(np.min(data[:,0].numpy()), np.min(data[:,2].numpy()))
    ymax = max(np.max(data[:,1].numpy()), np.max(data[:,3].numpy()))
    ymin = min(np.min(data[:,1].numpy()), np.min(data[:,3].numpy()))
    bounds = np.array([[xmin-1, xmax+1], [ymin-1, ymax+1], [xmin-1, xmax+1], [ymin-1, ymax+1]])

    def toy_feature_vec_func(s):
        """
        :param s: self, (T, 2)
        :param o: other, (T, 2)
        :return: features, (T, Df, 2)
        """
        corners = torch.tensor([[0, 0], [0, 8], [10, 0], [10,8]], dtype=torch.float64)
        return feature_direction_vec(s, corners)

    K = 3

    Df = 4
    lags = 3

    tran = LSTMBasedTransformation(K=K, D=D, Df=Df, feature_vec_func=toy_feature_vec_func, lags=lags)

    weights = tran.get_weights(data)

    # transformation
    feature_vecs_a = toy_feature_vec_func(data[:, 0:2])
    feature_vecs_b = toy_feature_vec_func(data[:, 2:4])
    feature_vecs = (feature_vecs_a, feature_vecs_b)

    transform_data = tran.transform(data, feature_vecs=feature_vecs)
    transform_data_2 = tran.transform(data)

    assert torch.allclose(transform_data, transform_data_2)


def test_model():
    torch.manual_seed(0)
    np.random.seed(0)

    T = 6
    D = 4

    # data = np.array([[1.0, 1.0, 1.0, 6.0], [3.0, 6.0, 8.0, 6.0],
    #                [4.0, 7.0, 8.0, 5.0], [6.0, 7.0, 5.0, 6.0], [8.0, 2.0, 6.0, 1.0]])
    data = np.random.randn(T, D)
    data = torch.tensor(data, dtype=torch.float64)

    xmax = max(np.max(data[:, 0].numpy()), np.max(data[:, 2].numpy()))
    xmin = min(np.min(data[:, 0].numpy()), np.min(data[:, 2].numpy()))
    ymax = max(np.max(data[:, 1].numpy()), np.max(data[:, 3].numpy()))
    ymin = min(np.min(data[:, 1].numpy()), np.min(data[:, 3].numpy()))
    bounds = np.array([[xmin - 1, xmax + 1], [ymin - 1, ymax + 1], [xmin - 1, xmax + 1], [ymin - 1, ymax + 1]])

    def toy_feature_vec_func(s):
        """
        :param s: self, (T, 2)
        :param o: other, (T, 2)
        :return: features, (T, Df, 2)
        """
        corners = torch.tensor([[0, 0], [0, 8], [10, 0], [10, 8]], dtype=torch.float64)
        return feature_direction_vec(s, corners)

    K = 3

    Df = 4
    lags = 1

    tran = LSTMBasedTransformation(K=K, D=D, Df=Df, feature_vec_func=toy_feature_vec_func, lags=lags)

    # observation
    obs = ARTruncatedNormalObservation(K=K, D=D, lags=lags, bounds=bounds, transformation=tran)

    # model
    model = HMM(K=K, D=D, observation=obs)

    print("calculating log likelihood")
    feature_vecs_a = toy_feature_vec_func(data[:-1, 0:2])
    feature_vecs_b = toy_feature_vec_func(data[:-1, 2:4])
    feature_vecs = (feature_vecs_a, feature_vecs_b)

    model.log_likelihood(data, feature_vecs=feature_vecs)

    # fit
    losses, _ = model.fit(data, optimizer=None, method="adam", num_iters=50, feature_vecs=feature_vecs)

    plt.figure()
    plt.plot(losses)
    plt.show()

    # most-likely-z
    print("Most likely z...")
    z = model.most_likely_states(data, feature_vecs=feature_vecs)


    # prediction

    if data.shape[0] <= 1000:
        data_to_predict = data
    else:
        data_to_predict = data[-1000:]

    print("0 step prediction")
    x_predict = k_step_prediction_for_lstm_based_model(model, z, data_to_predict, k=0, feature_vecs=feature_vecs)
    x_predict_err = np.mean(np.abs(x_predict - data_to_predict.numpy()), axis=0)


    k_step = 3
    print("{} step prediction".format(k_step))
    x_predict_k = k_step_prediction_for_lstm_based_model(model, z, data_to_predict, k=k_step)
    x_predict_k_err = np.mean(np.abs(x_predict_k - data_to_predict[k_step:].numpy()), axis=0)


    # samples
    print("sampling...")
    sample_T = 5
    lstm_states = {}
    sample_z, sample_x = model.sample(sample_T, lstm_states=lstm_states)



#test_tran()
test_model()
