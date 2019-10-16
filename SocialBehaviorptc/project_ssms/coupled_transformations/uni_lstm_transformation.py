import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence
import numpy as np

from ssm_ptc.utils import check_and_convert_to_tensor, set_param

from project_ssms.coupled_transformations.base_weighted_direction_transformation \
    import BaseWeightedDirectionTransformation
from project_ssms.coupled_transformations.lstm_based_transformation import MLPWeight


"""
one lstm
k - dynamics --> different output layers
weights(x_t) = acc_factor * sigmoid( output_layers ( LSTM(x_{t-1}, ..., x_{t-lags}) ) )
"""


def set_model_params(model1, params):
    i = 0
    for name, param in model1.state_dict().items():
        model1.state_dict()[name].copy_(params[i])
        i += 1


class UniLSTMTransformation(BaseWeightedDirectionTransformation):
    """
    weights of x_{t+1} depends on x_{t}, ...., x_{t-lags}
    """

    def __init__(self, K, D, Df, feature_vec_func, lags=1, dh=4, dhs=None, acc_factor=2):
        super(UniLSTMTransformation, self).__init__(K, D, Df, feature_vec_func, acc_factor, lags=lags)

        self.dh = dh
        self.dhs = dhs if dhs else []

        self.lstm = nn.LSTM(input_size=self.D, hidden_size=dh).double()

        self.acc_factor = acc_factor
        self.weight_nns = [MLPWeight(din=self.dh, dhs=self.dhs, dout=2*self.Df, acc_factor=self.acc_factor)
                           for _ in range(self.K)]
    @property
    def params(self):
        params = ()
        params += tuple(self.lstm.parameters())
        for weight_networks in self.weight_nns:
            params += tuple(weight_networks.parameters())
        return params

    @params.setter
    def params(self, values):
        # TODO
        pass

    def permute(self, perm):
        self.weight_nns = self.weight_nns[perm]

    def get_weights(self, inputs, **kwargs):
        """
        :param inputs: (T, D)
        :param kwargs:
        :return: a tuple of length 2, each is a set of weights of inputs, of shape (T-lags+1, K, Df)
        """

        packed_data = kwargs.get("packed_data", None)
        if packed_data is None:
            print("not using packed data memory")
            packed_data = get_packed_data(inputs, self.lags)

        T, D = inputs.shape
        assert D == self.D, \
            "inputs should have shape {}, instead of {}".format((T, self.D), (T, D))

        # first, transform to LSTM outputs
        _, (lstm_outputs, _) = self.lstm.forward(packed_data)
        assert lstm_outputs.shape == (1, T, self.dh), lstm_outputs.shape
        lstm_outputs = torch.squeeze(lstm_outputs, dim=0)

        # then, perform K different transformations
        # (T, K, 2*Df)
        weights = torch.stack([weight_nn.forward(lstm_outputs) for weight_nn in self.weight_nns], dim=1)
        assert weights.shape == (T, self.K, 2 * self.Df), \
            "weights_of_inputs should have shape ({}, {}, {}), instead of {}".format(T, self.K, 2*self.Df,
                                                                                     weights.shape)

        return weights[..., 0:self.Df], weights[..., self.Df:]

    def get_weights_condition_on_z(self, inputs, z, **kwargs):
        """

        :param inputs: (T_pre, 4)
        :param animal_idx: 0 or 1
        :param z: a scalar
        :param kwargs: a tuple of length 2, each is a set of weights of input, of shape (1, Df)
        :return:
        """

        T_pre, D = inputs.shape
        assert D == self.D
        assert T_pre <= self.lags, "T_pre={} must be smaller than or equal to lags={}".format(T_pre, self.lags)
        inputs = inputs[:, None, ]  # (T_pre, 1, 4)

        _, (h_t, c_t) = self.lstm.forward(inputs)
        assert h_t.shape == (1, 1, self.dh), h_t.shape
        h_t = torch.squeeze(h_t, dim=1)

        weights = self.weight_nns[z].forward(h_t)  # (1, 2*Df)
        assert weights.shape == (1, 2*self.Df), \
            "weights should have shape ({}, {}), instead of {}".format(1, 2*self.Df, weights.shape)

        return weights[..., 0:self.Df], weights[..., self.Df:]


def get_lagged_data(data, lags):
    """

    :param data: (T, D)
    :param lags: an integer
    :return: (lags, T-lags+1, D)
    """
    T, D = data.shape
    lagged_data = torch.stack([data[i:i + lags] for i in range(T - lags + 1)], dim=1)
    assert lagged_data.shape == (lags, T-lags+1, D), lagged_data.shape
    return lagged_data


def get_packed_data(data, lags):
    """

    :param data: (T, D)
    :param lags:
    :return: packed_sequence
    """
    T, D = data.shape
    pre_lagged_data = [data[0:t] for t in range(1, lags)]
    lagged_data = [data[i:i + lags] for i in range(T - lags + 1)]

    lagged_data = pre_lagged_data + lagged_data
    assert len(lagged_data) == T
    pack_seq = pack_sequence(lagged_data, enforce_sorted=False)
    return pack_seq
