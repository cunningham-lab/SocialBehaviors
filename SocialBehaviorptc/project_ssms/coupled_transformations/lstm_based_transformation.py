import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence
import torch.nn.functional as F
import numpy as np

from ssm_ptc.utils import check_and_convert_to_tensor, set_param

from project_ssms.coupled_transformations.base_weighted_direction_transformation \
    import BaseWeightedDirectionTransformation


class MLPWeight(nn.Module):
    def __init__(self, din, dhs, dout, acc_factor):
        super(MLPWeight, self).__init__()
        hidden_layers = []
        last_dh = din
        assert isinstance(dhs, list), "dhs must be a list, but is {}".format(type(dhs))
        # dhs=[] means no hidden layer and on,y one output layer
        for dh in dhs:
            hidden_layers.append(nn.Linear(last_dh, dh).double())
            last_dh = dh
        self.hidden_layers = nn.ModuleList(hidden_layers)

        self.output_layer = nn.Linear(last_dh, dout).double()
        self.acc_factor = acc_factor

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.acc_factor * torch.sigmoid(self.output_layer(x))
        return x


class LSTMBasedTransformation(BaseWeightedDirectionTransformation):
    """
    weights of x_{t+1} depends on x_{t}, ...., x_{t-lags}
    """

    def __init__(self, K, D, Df, feature_vec_func, lags=1, dh=4, dhs=None, acc_factor=2):
        super(LSTMBasedTransformation, self).__init__(K, D, Df, feature_vec_func, acc_factor, lags=lags)
        assert self.lags == 1
        self.dh = dh  # lstm dh
        self.dhs = dhs if dhs else []  # nn dhs
        self.lstm = nn.LSTM(input_size=self.D, hidden_size=dh).double()
        self.acc_factor = acc_factor
        self.weight_nns = [MLPWeight(din=self.dh, dhs=self.dhs, dout=2*self.Df, acc_factor=self.acc_factor)
                           for _ in range(self.K)]

    @property
    def params(self):
        params = ()
        params += tuple(self.lstm.parameters())

        for weight_nn in self.weight_nns:
            params += tuple(weight_nn.parameters())
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

        T, D = inputs.shape
        assert D == self.D, \
            "inputs should have shape {}, instead of {}".format((T, self.D), (T, D))

        outputs, _ = self.lstm.forward(input=inputs[:, None,])
        assert outputs.shape == (T, 1, self.dh)
        outputs = torch.squeeze(outputs, dim=1)

        weights = torch.stack([weight_nn.forward(outputs) for weight_nn in self.weight_nns], dim=1)
        assert weights.shape == (T, self.K, 2*self.Df), \
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

        lstm_states = kwargs.get("lstm_states", None)
        assert lstm_states is not None
        h_prev = lstm_states.get("h_t", None)
        c_prev = lstm_states.get("c_t", None)

        if h_prev is not None or c_prev is not None:
            _, (h_t, c_t) = self.lstm.forward(inputs[-1:,], (h_prev, c_prev))
        else:
            _, (h_t, c_t) = self.lstm.forward(inputs)

        assert h_t.shape == (1, 1, self.dh), h_t.shape
        assert c_t.shape == (1, 1, self.dh), c_t.shape

        weights = self.weight_nns[z].forward(h_t[0])
        assert weights.shape == (1, 2*self.Df), \
            "weights should have shape ({}, {}), instead of {}".format(1, 2*self.Df, weights.shape)

        lstm_states["h_t"] = h_t
        lstm_states["c_t"] = c_t

        return weights[..., 0:self.Df], weights[..., self.Df:]
