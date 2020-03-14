import torch
import numpy as np

from ssm_ptc.utils import check_and_convert_to_tensor, set_param

from project_ssms.coupled_transformations.base_weighted_direction_transformation \
    import BaseWeightedDirectionTransformation

# TODO: specify lags = 1
class WeightedGridTransformation(BaseWeightedDirectionTransformation):
    """
    Learnable parameters: weights of each grid vertex
    weights of any random point = a weighted combination of weights of all the grid vertices
    where the weights = softmax
    """
    def __init__(self, K, D, x_grids, y_grids, Df, feature_vec_func, acc_factor=2, beta=None, train_beta=True, lags=1):
        assert lags == 1, "lags should be 1 for weigthedgird with_noise"
        super(WeightedGridTransformation, self).__init__(K, D, Df, feature_vec_func, acc_factor, lags=lags)

        self.x_grids = check_and_convert_to_tensor(x_grids, dtype=torch.float64)  # [x_0, x_1, ..., x_m]
        self.y_grids = check_and_convert_to_tensor(y_grids, dtype=torch.float64)  # a list [y_0, y_1, ..., y_n]
        # shape: (d, n_gps)
        self.gridpoints = torch.tensor([(x_grid, y_grid) for x_grid in self.x_grids for y_grid in self.y_grids])
        self.gridpoints = torch.transpose(self.gridpoints, 0, 1)
        # number of basis grid points
        self.GP = self.gridpoints.shape[1]

        self.Ws = torch.rand(self.K, 2, self.GP, self.Df, dtype=torch.float64, requires_grad=True)

        # beta in the softmax expression
        if beta is None:
            beta = 0.1 * np.ones((self.K, 2))
        else:
            if isinstance(beta, (int, float)):
                beta = beta * np.ones((self.K, 2))
            else:
                assert beta.shape == (self.K, 2)
        self.beta = torch.tensor(beta, dtype=torch.float64, requires_grad=train_beta)

    @property
    def params(self):
        return self.Ws, self.beta

    @params.setter
    def params(self, values):
        self.Ws = set_param(self.Ws, values[0])
        self.beta = set_param(self.beta, values[1])

    def permute(self, perm):
        self.Ws = self.Ws[perm]
        self.beta = self.beta[perm]

    def get_weights(self, inputs, **kwargs):
        weights_a = self.get_weights_for_single_animal(inputs[:, 0:2], 0, **kwargs)
        weights_b = self.get_weights_for_single_animal(inputs[:, 2:4], 1, **kwargs)
        return weights_a, weights_b

    def get_weights_for_single_animal(self, inputs, animal_idx, **kwargs):
        """
        w(x) = \sum_{g=1}^G softmax(-\beta d_g) w_g
        :param inputs: (T, 2)
        :param animal_idx: 0 or 1
        :param kwargs:
        :return: weights of inputs, shape (T, K, Df)
        """
        T, d = inputs.shape
        assert d == self.d, "inputs should have shape {}, instead of {}".format((T, self.d), (T, d))

        distances_key = "distances_a" if animal_idx == 0 else "distances_b"
        distances_s = kwargs.get(distances_key, None)
        if distances_s is None:
            #print("not providing distance memories")
            distances_s = pairwise_dist(inputs, self.gridpoints.t())  # (T, n_gps)
        assert distances_s.shape == (T, self.GP), \
            "distances_s should have shape {}, instead of {}".format((T, self.GP), distances_s.shape)

        # coefficients of the grid weights: (T, K, n_gps)
        # (K, 1) * (T, 1, n_gps) -> (T, K, n_gps)
        coef = torch.softmax(-torch.matmul(self.beta[:, animal_idx, None], distances_s[:, None]), dim=-1)
        assert coef.shape == (T, self.K, self.GP), \
            "coef should have shape {}, instead of {}".format((T, self.K, self.GP), coef.shape)

        # (T, K, 1, n_gps) * (K, n_gps, Df) --> (T, K, 1, Df)
        weigths_of_inputs = torch.matmul(coef[:,:, None, ], self.Ws[:, animal_idx])
        assert weigths_of_inputs.shape == (T, self.K, 1, self.Df), \
            "weights_of_inputs should ahve shape {}, instead of {}".format((T, self.K, 1, self.Df),
                                                                           weigths_of_inputs.shape)
        weigths_of_inputs = torch.squeeze(weigths_of_inputs, dim=2)

        weigths_of_inputs = self.acc_factor * torch.sigmoid(weigths_of_inputs)

        return weigths_of_inputs

    def get_weights_condition_on_z(self, inputs, z, **kwargs):
        weights_a = self.get_weights_condition_on_z_for_single_animal(inputs[:, 0:2], animal_idx=0, z=z, **kwargs)
        weights_b = self.get_weights_condition_on_z_for_single_animal(inputs[:, 2:4], animal_idx=1, z=z, **kwargs)
        return weights_a, weights_b

    def get_weights_condition_on_z_for_single_animal(self, inputs, animal_idx, z, **kwargs):
        """

        :param inputs: (T_pre, 2)
        :param animal_idx: 0 or 1
        :param z: a scalar
        :param kwargs: weights of input, shape (1, Df)
        :return:
        """

        _, d = inputs.shape
        assert d == self.d

        distance_key = "distance_a" if animal_idx == 0 else "distance_b"
        distance_s = kwargs.get(distance_key, None)
        if distance_s is None:
            #print("not providing distance memory")
            distance_s = pairwise_dist(inputs[-1:], self.gridpoints.t())  #  (1, n_gps)
        assert distance_s.shape == (1, self.GP), \
            "distance_a should have shape {}, instead of {}".format((1, self.GP), distance_s.shape)

        # coefficients of the grid weights: (1, n_gps)
        # () * (1, n_gps) -> (1, n_gps)
        coef = torch.softmax(-self.beta[z, animal_idx] * distance_s, dim=-1)
        assert coef.shape == (1, self.GP), "coef should have shape {}, instead of {}".format((1, self.GP), coef.shape)

        # (1, n_gps) * (n_gps, Df) --> (1, Df)
        weight_of_input = torch.matmul(coef, self.Ws[z, animal_idx])
        assert weight_of_input.shape == (1, self.Df), \
            "weight_of_input should have shape {}, instead of {}".format((1, self.Df), weight_of_input.shape)

        weight_of_input = self.acc_factor * torch.sigmoid(weight_of_input)

        return weight_of_input


def pairwise_dist(points_a, points_b):
    """

    :param points_a: (n1, 2)
    :param points_b: (n2, 2)
    :return: pariwise distance d of shape (n1, n2), where d_ij = distance(points_a[i], points_b[j])
    """
    n1, d = points_a.shape
    assert d == 2, "points_a should have shape {}, instead of {}".format((n1, 2), (n1, d))

    n2, d = points_b.shape
    assert d == 2, "points_b should have shape {}, instead of {}".format((n2, 2), (n2, d))

    x_dist = points_a[:, None, 0] - points_b[None, :, 0]
    assert x_dist.shape == (n1, n2)

    y_dist = points_a[:, None, 1] - points_b[None, :, 1]

    dist = torch.sqrt(x_dist**2 + y_dist**2)
    assert dist.shape == (n1, n2)
    return dist


def test_dist():
    points_a = torch.tensor([[0, 0], [1, 1]], dtype=torch.float64)
    points_b = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.float64)

    correct_pdist = torch.tensor([[0, np.sqrt(2), 2*np.sqrt(2)], [np.sqrt(2), 0, np.sqrt(2)]],
                                 dtype=torch.float64)

    pdist = pairwise_dist(points_a, points_b)

    assert torch.all(pdist == correct_pdist)


if __name__ == '__main__':
    test_dist()
    print("yes")





