import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence
import numpy as np

from ssm_ptc.utils import check_and_convert_to_tensor, set_param

from project_ssms.coupled_transformations.base_weighted_direction_transformation \
    import BaseWeightedDirectionTransformation


class LSTMWeight(nn.Module):
    def __init__(self, din, dh, dout, acc_factor):
        super(LSTMWeight, self).__init__()
        self.dh = dh
        self.dout = dout

        self.lstm = nn.LSTM(input_size=din, hidden_size=dh).double()

        self.output_layer = nn.Linear(dh, dout).double()
        self.acc_factor = acc_factor

    def forward(self, x):
        """

        :param x: (seq_length, batch_size, din) or packed_sequence
        :return: f(x): (batch_size, dout)
        """
        #T, bs, din = x.shape
        _, (h_t, c_t) = self.lstm.forward(x)
        #assert h_t.shape == (1, bs, self.dh)
        x = self.acc_factor * torch.sigmoid(self.output_layer(h_t[0]))
        #assert x.shape == (bs, self.dout)
        return x


def set_model_params(model1, params):
    i = 0
    for name, param in model1.state_dict().items():
        model1.state_dict()[name].copy_(params[i])
        i += 1


class LSTMTransformation(BaseWeightedDirectionTransformation):
    """
    weights of x_{t+1} depends on x_{t}, ...., x_{t-lags}
    """

    def __init__(self, K, D, Df, feature_vec_func, lags=1, dh=4, acc_factor=2):
        super(LSTMTransformation, self).__init__(K, D, Df, feature_vec_func, acc_factor, lags=lags)

        self.weight_networks = [LSTMWeight(self.D, dh, 2*self.Df, self.acc_factor) for _ in range(self.K)]

    @property
    def params(self):
        params = ()
        for weight_networks in self.weight_networks:
            params += tuple(weight_networks.parameters())
        return params

    @params.setter
    def params(self, values):
        self.Ws = set_param(self.Ws, values[0])
        self.beta = set_param(self.beta, values[1])

    def permute(self, perm):
        self.Ws = self.Ws[perm]
        self.beta = self.beta[perm]

    def get_weights(self, inputs, **kwargs):
        """
        :param inputs: (T, D)
        :param kwargs:
        :return: a tuple of length 2, each is a set of weights of inputs, of shape (T-lags+1, K, Df)
        """

        packed_data = kwargs.get("packed_data", None)
        if packed_data is None:
            #print("not using packed data memory")
            packed_data = get_packed_data(inputs, self.lags)

        T, D = inputs.shape
        assert D == self.D, \
            "inputs should have shape {}, instead of {}".format((T, self.D), (T, D))

        # (T, K, 2*Df)
        weights_of_inputs = \
            torch.stack([weight_network.forward(packed_data) for weight_network in self.weight_networks], dim=1)
        assert weights_of_inputs.shape == (T, self.K, 2 * self.Df), \
            "weights_of_inputs should have shape ({}, {}, {}), instead of {}".format(T, self.K, 2*self.Df,
                                                                                     weights_of_inputs.shape)

        return weights_of_inputs[..., 0:self.Df], weights_of_inputs[..., self.Df:]

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

        weights_of_inputs = self.weight_networks[z].forward(inputs)  # (1, 2*Df)
        assert weights_of_inputs.shape == (1, 2*self.Df), \
            "weights_of_inputs should have shape ({}, {}), instead of {}".format(1, 2*self.Df, weights_of_inputs.shape)

        return weights_of_inputs[..., 0:self.Df], weights_of_inputs[..., self.Df:]


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
