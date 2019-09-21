import torch
import numpy as np

from project_ssms.coupled_transformations.lineargrid_transformation import LinearGridTransformation,\
    one_d_interpolation, two_d_interpolation
from project_ssms.feature_funcs import f_corner_vec_func, feature_direction_vec
from project_ssms.ar_truncated_normal_observation import ARTruncatedNormalObservation
from project_ssms.utils import k_step_prediction_for_lineargrid_model

from ssm_ptc.models.hmm import HMM
from ssm_ptc.utils import k_step_prediction

import matplotlib.pyplot as plt


def test_one_d_interpolation():
    # batch case
    x = torch.tensor([[1.0], [2.0]])  # (T, 1)

    x1 = torch.tensor([[0.0], [0.0]])  # (T, 1)
    x2 = torch.tensor([[4.0], [6.0]])  # (T, 1)

    f_x1 = torch.tensor([[[0.0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])  # (T, K, Df)
    f_x2 = torch.tensor([[[4.0, 8.0, 12], [16, 32, 64]], [[9, 12, 15], [18, 21, 24]]])  # (T, K, Df)

    f_x = one_d_interpolation(x, x1, x2, f_x1, f_x2)  # (T, K, Df)

    true_f_x = torch.tensor([[[1.0, 2, 3], [4, 8, 16]], [[3, 4, 5], [6, 7, 8]]])
    assert torch.all(torch.eq(f_x, true_f_x))

    # single point case
    x = torch.tensor([1.0])  # (T, 1)

    x1 = torch.tensor([0.0])  # (T, 1)
    x2 = torch.tensor([4.0])  # (T, 1)

    f_x1 = torch.tensor([[0.0, 0, 0], [0, 0, 0]])  # (K, Df)
    f_x2 = torch.tensor([[4.0, 8.0, 12], [16, 32, 64]])  # (K, Df)

    f_x = one_d_interpolation(x, x1, x2, f_x1, f_x2)  # (K, Df)

    true_f_x = torch.tensor([[1.0, 2, 3], [4, 8, 16]])
    assert torch.all(torch.eq(f_x, true_f_x))


def test_two_d_interpolation():
    # batch case
    points = torch.tensor([[1.0, 2], [11, 12]])  # (T, 2)

    Q11 = torch.tensor([[0.0, 0], [10.0, 10]])  # (T, 2)
    Q22 = torch.tensor([[4.0, 4], [14, 14]])

    f_Q11 = torch.tensor([[[0.0, 0], [0, 0]], [[0, 0], [8, 8]]])  # (T, K, Df)
    f_Q21 = torch.tensor([[[4.0, 4], [8, 8]], [[8, 8], [16, 16]]])

    f_Q12 = torch.tensor([[[8.0, 8], [16, 16]], [[16, 16], [32, 32]]])
    f_Q22 = torch.tensor([[[12.0, 12], [24, 24]], [[24, 24], [48, 48]]])

    f_points = two_d_interpolation(points, Q11, Q22, f_Q11, f_Q12, f_Q21, f_Q22)

    true_f_points = torch.tensor([[[5.0, 5], [10, 10]], [[10, 10], [23, 23]]])  # (T, K, Df)
    assert torch.all(torch.eq(f_points, true_f_points))

    # single point case
    points = torch.tensor([1.0, 2])  # (2, )

    Q11 = torch.tensor([0.0, 0])  # (2, )
    Q22 = torch.tensor([4.0, 4])

    f_Q11 = torch.tensor([[0.0, 0], [0, 0]])  # (K, Df)
    f_Q21 = torch.tensor([[4.0, 4], [8, 8]])

    f_Q12 = torch.tensor([[8.0, 8], [16, 16]])
    f_Q22 = torch.tensor([[12.0, 12], [24, 24]])

    f_points = two_d_interpolation(points, Q11, Q22, f_Q11, f_Q12, f_Q21, f_Q22)

    true_f_points = torch.tensor([[5.0, 5], [10, 10]])  # (K, Df)
    assert torch.all(torch.eq(f_points, true_f_points))


def test_get_grid_point_idx():
    # single point case
    x_grids = np.array([0, 1, 2, 3, 4])
    y_grids = np.array([0, 1, 2, 3])
    tran = LinearGridTransformation(K=4, D=4, x_grids=x_grids, y_grids=y_grids,
                                    Df=4, feature_vec_func=f_corner_vec_func)

    point = torch.tensor([0.5, 0.5], dtype=torch.float64)
    idx = tran.get_gridpoints_idx_for_single(point)  # (GP, 4)

    true_idx_0 = torch.zeros((20, 4), dtype=torch.float64)
    true_idx_0[0, 0] = 1
    true_idx_0[1, 1] = 1
    true_idx_0[4, 2] = 1
    true_idx_0[5, 3] = 1
    assert torch.all(torch.eq(idx, true_idx_0))

    point = torch.tensor([2.5, 1.9], dtype=torch.float64)
    idx = tran.get_gridpoints_idx_for_single(point)

    true_idx_1 = torch.zeros((20, 4), dtype=torch.float64)
    true_idx_1[9, 0] = 1
    true_idx_1[10, 1] = 1
    true_idx_1[13, 2] = 1
    true_idx_1[14, 3] = 1
    assert torch.all(torch.eq(idx, true_idx_1))

    point = torch.tensor([0, 0], dtype=torch.float64)
    idx = tran.get_gridpoints_idx_for_single(point)

    true_idx_2 = torch.zeros((20, 4), dtype=torch.float64)
    true_idx_2[0, 0] = 1
    true_idx_2[1, 1] = 1
    true_idx_2[4, 2] = 1
    true_idx_2[5, 3] = 1
    assert torch.all(torch.eq(idx, true_idx_2))

    point = torch.tensor([4, 3], dtype=torch.float64)
    idx = tran.get_gridpoints_idx_for_single(point)

    true_idx_3 = torch.zeros((20, 4), dtype=torch.float64)
    true_idx_3[14, 0] = 1
    true_idx_3[15, 1] = 1
    true_idx_3[18, 2] = 1
    true_idx_3[19, 3] = 1
    assert torch.all(torch.eq(idx, true_idx_3))

    # batch case
    points = torch.tensor([[0.5, 0.5], [2.5, 1.9], [0.0, 0.0], [4.0, 3.0]], dtype=torch.float64)
    idx = tran.get_gridpoints_idx_for_batch(points)

    # (T, GP, 4)
    true_idx = torch.stack([true_idx_0, true_idx_1, true_idx_2, true_idx_3], dim=0)
    assert torch.all(torch.eq(idx, true_idx))


def test_weights():

    x_grids = np.array([0, 1, 2, 3, 4])
    y_grids = np.array([0, 1, 2, 3])

    tran = LinearGridTransformation(K=4, D=4, x_grids=x_grids, y_grids=y_grids,
                                    Df=4, feature_vec_func=f_corner_vec_func)

    # single point case
    point = torch.tensor([0.5, 0.5], dtype=torch.float64)
    idx = tran.get_gridpoints_idx_for_single(point)  # (GP, 4)
    # (d, GP) * (GP, 2) --> (d, 2)
    grid_points = torch.matmul(tran.gridpoints, idx[:, [0, -1]])

    true_grid_points = torch.tensor([[0, 1], [0, 1]], dtype=torch.float64)   # (2, 2)
    assert torch.all(torch.eq(grid_points, true_grid_points))

    weights = tran.get_weights_for_single(point=point, animal_idx=0, grid_points=grid_points, grid_points_idx=idx, z=1)

    grid_points_weights = tran.Ws[1, 0, [0, 1, 4, 5], :]  # (4, Df)
    true_weights = torch.mean(grid_points_weights, dim=0, keepdim=True)  # (1, Df)

    assert torch.allclose(weights, true_weights)
    #assert torch.all(torch.eq(weights, true_weights))

    # batch case
    points = torch.tensor([[0.5, 0.5], [2, 2], [4, 3]], dtype=torch.float64)
    T = points.shape[0]
    idx = tran.get_gridpoints_idx_for_batch(points)  # (T, GP, 4)
    assert idx.shape == (T, tran.GP, 4)

    grid_points = torch.matmul(tran.gridpoints, idx[:, :, [0,-1]])  # (T, d, 2)

    true_grid_points = torch.tensor([[[0, 1], [0, 1]], [[1, 2], [1, 2]], [[3, 4], [2, 3]]], dtype=torch.float64)
    assert torch.all(torch.eq(grid_points, true_grid_points))

    weights = tran.get_weights_for_batch(points=points, animal_idx=0, grid_points=grid_points, grid_points_idx=idx)
    assert weights.shape == (T, tran.K, tran.Df)

    true_weights_0 = tran.Ws[:, 0, [0, 1, 4, 5], :]  # (K, 4, Df)
    true_weights_0 = torch.mean(true_weights_0, dim=1)  # (K, Df)
    true_weights_1 = tran.Ws[:, 0, 10]  # (K, Df)
    true_weights_2 = tran.Ws[:, 0, 19]  # (K, Df)

    true_weights = torch.stack([true_weights_0, true_weights_1, true_weights_2], dim=0) # (T, K, Df)
    assert torch.allclose(weights, true_weights)


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

    tran = LinearGridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, Df=Df,
                                    feature_vec_func=toy_feature_vec_func)

    assert tran.GP == 9

    # transformation
    transformed_data = tran.transform(data)

    # calculate memory
    grid_points_idx_a = tran.get_gridpoints_idx_for_batch(data[:, 0:2])  # (T, GP, 4)
    grid_points_idx_b = tran.get_gridpoints_idx_for_batch(data[:, 2:4])  # (T, GP, 4)
    grid_points_idx = (grid_points_idx_a, grid_points_idx_b)

    gridpoints_a = tran.get_gridpoints_for_batch(grid_points_idx_a)  # (T, d, 2)
    gridpoints_b = tran.get_gridpoints_for_batch(grid_points_idx_b)  # (T, d, 2)

    feature_vecs_a = toy_feature_vec_func(data[:, 0:2])
    feature_vecs_b = toy_feature_vec_func(data[:, 2:4])
    feature_vecs = (feature_vecs_a, feature_vecs_b)

    transformed_data_2 = tran.transform(data, gridpoints=(gridpoints_a, gridpoints_b),
                                        gridpoints_idx=grid_points_idx, feature_vecs=feature_vecs)

    assert torch.all(torch.eq(transformed_data, transformed_data_2))

    # transform condition on z
    transform_data_condition_on_z = tran.transform_condition_on_z(0, data)

    # calculate memory
    grid_points_idx_last = (grid_points_idx_a[-1], grid_points_idx_b[-1])
    gridpoints_last = (gridpoints_a[-1], gridpoints_b[-1])
    feature_vec_last = (feature_vecs_a[-1:], feature_vecs_b[-1:])
    transform_data_condition_on_z_2 = tran.transform_condition_on_z(0, data, gridpoints=gridpoints_last,
                                                                    gridpoints_idx=grid_points_idx_last,
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
    tran = LinearGridTransformation(K=K, D=D, x_grids=x_grids, y_grids=y_grids, Df=Df,
                                    feature_vec_func=toy_feature_vec_func)
    obs = ARTruncatedNormalObservation(K=K, D=D, M=0, lags=1, bounds=bounds, transformation=tran)

    model = HMM(K=K, D=D, M=M, transition="stationary", observation=obs)
    model.observation.mus_init = data[0] * torch.ones(K, D, dtype=torch.float64)

    # calculate memory
    gridpoints_idx_a = tran.get_gridpoints_idx_for_batch(data[:-1, 0:2])
    gridpoints_idx_b = tran.get_gridpoints_idx_for_batch(data[:-1, 2:4])
    gridpoints_a = tran.get_gridpoints_for_batch(gridpoints_idx_a)
    gridpoints_b = tran.get_gridpoints_for_batch(gridpoints_idx_b)
    feature_vecs_a = toy_feature_vec_func(data[:-1, 0:2])
    feature_vecs_b = toy_feature_vec_func(data[:-1, 2:4])

    gridpoints_idx = (gridpoints_idx_a, gridpoints_idx_b)
    gridpoints = (gridpoints_a, gridpoints_b)
    feature_vecs = (feature_vecs_a, feature_vecs_b)

    # fit
    losses, opt = model.fit(data, optimizer=None, method='adam', num_iters=100, lr=0.01,
                            pbar_update_interval=10,
                            gridpoints=gridpoints,
                            gridpoints_idx=gridpoints_idx, feature_vecs=feature_vecs)

    plt.figure()
    plt.plot(losses)
    plt.show()

    # most-likely-z
    print("Most likely z...")
    z = model.most_likely_states(data, gridpoints_idx=gridpoints_idx, feature_vecs=feature_vecs)

    # prediction
    print("0 step prediction")
    if data.shape[0] <= 1000:
        data_to_predict = data
    else:
        data_to_predict = data[-1000:]
    x_predict = k_step_prediction_for_lineargrid_model(model, z, data_to_predict,
                                                       gridpoints_idx=gridpoints_idx, feature_vecs=feature_vecs)
    x_predict_err = np.mean(np.abs(x_predict - data_to_predict.numpy()), axis=0)

    print("2 step prediction")
    x_predict_2 = k_step_prediction(model, z, data_to_predict, k=2)
    x_predict_2_err = np.mean(np.abs(x_predict_2 - data_to_predict[2:].numpy()), axis=0)

    # samples
    sample_T = 5
    sample_z, sample_x = model.sample(sample_T)




test_one_d_interpolation()
test_two_d_interpolation()
test_get_grid_point_idx()
test_weights()
test_tran()
test_model()
