import torch
import numpy as np

from project_ssms.coupled_transformations.weightedgrid_transformation import WeightedGridTransformation, pairwise_dist
from project_ssms.feature_funcs import feature_direction_vec
from project_ssms.ar_truncated_normal_observation import ARTruncatedNormalObservation
from project_ssms.utils import k_step_prediction_for_weightedgrid_model

from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import k_step_prediction

import matplotlib.pyplot as plt


def test_tran():
    torch.manual_seed(0)
    np.random.seed(0)

    T = 5
    x_grids = np.array([0.0, 5.0, 10.0])
    y_grids = np.array([0.0, 4.0, 8.0])

    data = np.array([[1.0, 1.0, 1.0, 6.0], [3.0, 6.0, 8.0, 6.0],
                     [4.0, 7.0, 8.0, 5.0], [6.0, 7.0, 5.0, 6.0], [8.0, 2.0, 6.0, 1.0]])
    data = torch.tensor(data, dtype=torch.float64)

    def toy_feature_vec_func(s):
        """
        :param s: self, (T, 2)
        :param o: other, (T, 2)
        :return: features, (T, Df, 2)
        """
        corners = torch.tensor([[0, 0], [0, 8], [10, 0], [10,8]], dtype=torch.float64)
        return feature_direction_vec(s, corners)

    K = 3
    D = 4

    Df = 4

    tran = WeightedGridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, Df=Df,
                                    feature_vec_func=toy_feature_vec_func)

    assert tran.GP == 9

    # transformation
    transformed_data = tran.transform(data)

    # calculate memory
    distances_a = pairwise_dist(data[:, 0:2], tran.gridpoints.t())
    distances_b = pairwise_dist(data[:, 2:4], tran.gridpoints.t())

    feature_vecs_a = toy_feature_vec_func(data[:, 0:2])
    feature_vecs_b = toy_feature_vec_func(data[:, 2:4])
    feature_vecs = (feature_vecs_a, feature_vecs_b)

    transformed_data_2 = tran.transform(data, distances_a=distances_a, distances_b=distances_b, feature_vecs=feature_vecs)

    assert torch.all(torch.eq(transformed_data, transformed_data_2))

    # transform condition on z
    transform_data_condition_on_z = tran.transform_condition_on_z(0, data)

    # calculate memory
    feature_vec_last = (feature_vecs_a[-1:], feature_vecs_b[-1:])
    transform_data_condition_on_z_2 = tran.transform_condition_on_z(0, data, distance_a=distances_a[-1:],
                                                                    distance_b=distances_b[-1:],
                                                                    feature_vec=feature_vec_last)

    assert torch.all(torch.eq(transform_data_condition_on_z, transform_data_condition_on_z_2))


def test_model():
    torch.manual_seed(0)
    np.random.seed(0)

    T = 5
    x_grids = np.array([0.0, 5.0, 10.0])
    y_grids = np.array([0.0, 4.0, 8.0])

    data = np.array([[1.0, 1.0, 1.0, 6.0], [3.0, 6.0, 8.0, 6.0],
                     [4.0, 7.0, 8.0, 5.0], [6.0, 7.0, 5.0, 6.0], [8.0, 2.0, 6.0, 1.0]])
    data = torch.tensor(data, dtype=torch.float64)

    def toy_feature_vec_func(s):
        """
        :param s: self, (T, 2)
        :param o: other, (T, 2)
        :return: features, (T, Df, 2)
        """
        corners = torch.tensor([[0, 0], [0, 8], [10, 0], [10, 8]], dtype=torch.float64)
        return feature_direction_vec(s, corners)

    K = 3
    D = 4
    M = 0

    Df = 4

    bounds = np.array([[0.0, 10.0], [0.0, 8.0], [0.0, 10.0], [0.0, 8.0]])
    tran = WeightedGridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, Df=Df,
                                    feature_vec_func=toy_feature_vec_func)
    obs = ARTruncatedNormalObservation(K=K, D=D, M=0, lags=1, bounds=bounds, transformation=tran)

    model = HMM(K=K, D=D, M=M, transition="stationary", observation=obs)
    model.observation.mus_init = data[0] * torch.ones(K, D, dtype=torch.float64)

    # calculate memory
    distances_a = pairwise_dist(data[:-1, 0:2], tran.gridpoints.t())
    distances_b = pairwise_dist(data[:-1, 2:4], tran.gridpoints.t())

    feature_vecs_a = toy_feature_vec_func(data[:-1, 0:2])
    feature_vecs_b = toy_feature_vec_func(data[:-1, 2:4])

    feature_vecs = (feature_vecs_a, feature_vecs_b)

    # fit
    losses, opt = model.fit(data, optimizer=None, method='adam', num_iters=100, lr=0.01,
                            pbar_update_interval=10,
                            distances_a=distances_a, distances_b=distances_b, feature_vecs=feature_vecs)

    plt.figure()
    plt.plot(losses)
    plt.show()

    # most-likely-z
    print("Most likely z...")
    z = model.most_likely_states(data, distances_a=distances_a, distances_b=distances_b, feature_vecs=feature_vecs)

    # prediction
    print("0 step prediction")
    if data.shape[0] <= 1000:
        data_to_predict = data
    else:
        data_to_predict = data[-1000:]
    x_predict = k_step_prediction_for_weightedgrid_model(model, z, data_to_predict,
                                                      distances_a=distances_a, distances_b=distances_b, feature_vecs=feature_vecs)
    x_predict_err = np.mean(np.abs(x_predict - data_to_predict.numpy()), axis=0)

    print("2 step prediction")
    x_predict_2 = k_step_prediction(model, z, data_to_predict, k=2)
    x_predict_2_err = np.mean(np.abs(x_predict_2 - data_to_predict[2:].numpy()), axis=0)

    # samples
    sample_T = 5
    sample_z, sample_x = model.sample(sample_T)



#test_tran()
test_model()
