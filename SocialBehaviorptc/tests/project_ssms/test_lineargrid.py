import torch
import numpy as np

from project_ssms.coupled_transformations.lineargrid_transformation import LinearGridTransformation,\
    one_d_interpolation, two_d_interpolation
from project_ssms.feature_funcs import f_corner_vec_func, feature_direction_vec


def test_one_d_interpolation():
    x = torch.tensor([[1.0]])  # (T, 1)

    x1 = torch.tensor([[0.0]])  # (T, 1)
    x2 = torch.tensor([[4.0]])  # (T, 1)

    f_x1 = torch.tensor([[[0.0, 0, 0], [0, 0, 0]]])  # (T, K, Df)
    f_x2 = torch.tensor([[[4.0, 8.0, 12], [16, 32, 64]]])  #  (T, K, Df)

    f_x = one_d_interpolation(x, x1, x2, f_x1, f_x2)

    true_f_x = torch.tensor([[[1.0, 2, 3], [4, 8, 16]]])
    assert torch.all(torch.eq(f_x, true_f_x))


def test_two_d_interpolation():
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


def test_get_grid_point_idx():
    x_grids = np.array([0, 1, 2, 3, 4])
    y_grids = np.array([0, 1, 2, 3])
    tran = LinearGridTransformation(K=4, D=4, x_grids=x_grids, y_grids=y_grids,
                                    Df=4, feature_vec_func=f_corner_vec_func)

    point = torch.tensor([0.5, 0.5], dtype=torch.float64)
    idx = tran.get_grid_point_idx(point, x_grids, y_grids)
    assert idx == [0, 1, 4, 5]

    point = torch.tensor([2.5, 1.9], dtype=torch.float64)
    idx = tran.get_grid_point_idx(point, x_grids, y_grids)
    assert idx == [9, 10, 13, 14]

    point = torch.tensor([0, 0], dtype=torch.float64)
    idx = tran.get_grid_point_idx(point, x_grids, y_grids)
    assert idx == [0, 1, 4, 5]

    point = torch.tensor([4, 3], dtype=torch.float64)
    idx = tran.get_grid_point_idx(point, x_grids, y_grids)
    assert idx == [14, 15, 18, 19]


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

    feature_vecs_a = toy_feature_vec_func(data[:, 0:2])
    feature_vecs_b = toy_feature_vec_func(data[:, 2:4])
    feature_vecs = (feature_vecs_a, feature_vecs_b)

    transformed_data_2 = tran.transform(data, memory_kwargs={"feature_vecs": feature_vecs})

    assert torch.all(torch.eq(transformed_data, transformed_data_2))

    # transform condition on z
    transform_data_condition_on_z = tran.transform_condition_on_z(0, data)

    feature_vec_last = (feature_vecs_a[-1:], feature_vecs_b[-1:])
    transform_data_condition_on_z_2 = tran.transform_condition_on_z(0, data,
                                                                  memory_kwargs={"feature_vec": feature_vec_last})

    assert torch.all(torch.eq(transform_data_condition_on_z, transform_data_condition_on_z_2))


test_one_d_interpolation()
test_two_d_interpolation()
test_get_grid_point_idx()
test_tran()
