import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.utils import check_and_convert_to_tensor, set_param


ACTIVATION_DICT = dict(relu=torch.nn.functional.relu)


class StateDependentWeight(nn.Module):
    def __init__(self, din, dhs, dout, acc_factor):
        super(StateDependentWeight, self).__init__()
        hidden_layers = []
        last_dh = din
        for dh in dhs:
            hidden_layers.append(nn.Linear(last_dh, dh).double())
            last_dh = dh
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(dhs[-1], dout).double()
        self.acc_factor = acc_factor

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.acc_factor * torch.sigmoid(self.output_layer(x))
        return x


def set_model_params(model1, params):
    i = 0
    for name, param in model1.state_dict().items():
        model1.state_dict()[name].copy_(params[i])
        i += 1


class SpaceDependentTransformation(BaseTransformation):
    """
        x^{self}_t \sim x^{self}_{t-1} + acc_factor * [ \sum_{i=1}^{Df} sigmoid( f_k(self) ) d_i (self)]
    """
    def __init__(self, K, D, Df, feature_vec_func, dhs, acc_factor):
        super(SpaceDependentTransformation, self).__init__(K, D)
        self.d = int(self.D / 2)

        self.Df = Df
        self.feature_vec_func = feature_vec_func

        self.dhs = dhs
        self.acc_factor = acc_factor

        self.weights_a = [StateDependentWeight(self.d, dhs, self.Df, self.acc_factor) for _ in range(self.K)]
        self.weights_b = [StateDependentWeight(self.d, dhs, self.Df, self.acc_factor) for _ in range(self.K)]

    @property
    def params(self):
        params = ()
        # TODO: test this
        for w_a, w_b in zip(self.weights_a, self.weights_b):
            params = params + tuple(w_a.parameters())  # 2 * number of layers (hidden + output)
            params = params + tuple(w_b.parameters())
        return params

    @params.setter
    def params(self, values):
        # TODO: test this
        i = 0
        model_param_size = 2*(len(self.dhs) + 1)

        for k in range(len(self.weights_a)):
            set_model_params(self.weights_a[k], values[i:i+model_param_size])
            set_model_params(self.weights_b[k], values[i+model_param_size:i+2*model_param_size])
            i = i + 2*model_param_size

    def permute(self, perm):
        self.weights_a = self.weights_a[perm]
        self.weights_b = self.weights_b[perm]

    def transform(self, inputs, memory_kwargs_a=None, memory_kwargs_b=None):
        """
        Transform based on the current grid
        :param inputs: (T, 4)
        :param memory_kwargs_a:
        :param memory_kwargs_b:
        :return: outputs (T, K, 4)
        """

        inputs = check_and_convert_to_tensor(inputs)
        T, _ = inputs.shape

        memory_kwargs_a = memory_kwargs_a or {}
        memory_kwargs_b = memory_kwargs_b or {}

        feature_vecs_a = memory_kwargs_a.get("feature_vecs", None)
        feature_vecs_b = memory_kwargs_b.get("feature_vecs", None)

        if feature_vecs_a is None:
            feature_vecs_a = self.feature_vec_func(inputs[:, 0:2])
        if feature_vecs_b is None:
            feature_vecs_b = self.feature_vec_func(inputs[:, 2:4])

        assert feature_vecs_a.shape == (T, self.Df, self.d), \
            "Feature vec a shape is " + str(feature_vecs_a.shape) \
            + ". It should have shape ({}, {}, {}).".format(T, self.Df, self.d)
        assert feature_vecs_b.shape == (T, self.Df, self.d), \
            "Feature vec b shape is " + str(feature_vecs_a.shape) \
            + ". It should have shape ({}, {}, {}).".format(T, self.Df, self.d)

        # (T, K, Df)
        weights_a = torch.stack([w_a.forward(inputs[:, 0:2]) for w_a in self.weights_a], dim=1)
        weights_b = torch.stack([w_b.forward(inputs[:, 2:4]) for w_b in self.weights_b], dim=1)
        assert weights_a.shape == (T, self.K, self.Df)
        assert weights_b.shape == (T, self.K, self.Df)

        # (T, K, Df) * (T, Df, d) -> (T, K, d)
        out_a = torch.matmul(weights_a, feature_vecs_a)
        assert out_a.shape == (T, self.K, self.d)
        out_b = torch.matmul(weights_b, feature_vecs_b)
        assert out_b.shape == (T, self.K, self.d)

        out = torch.cat((out_a, out_b), dim=-1) # (T, K, D)
        out = inputs[:, None, ] + out
        assert out.shape == (T, self.K, self.D)

        return out

    def transform_condition_on_z(self, z, inputs, memory_kwargs_a=None, memory_kwargs_b=None):
        """

        :param z: an integer
        :param inputs: (T_pre, D)
        :param memory_kwargs_a:
        :param memory_kwargs_b:
        :return: (D, )
        """
        memory_kwargs_a = memory_kwargs_a or {}
        memory_kwargs_b = memory_kwargs_b or {}

        feature_vec_a = memory_kwargs_a.get("feature_vec", None)
        feature_vec_b = memory_kwargs_b.get("feature_vec", None)

        if feature_vec_a is None:
            feature_vec_a = torch.squeeze(self.feature_vec_func(inputs[-1:, 0:2]), dim=0)
        if feature_vec_b is None:
            feature_vec_b = torch.squeeze(self.feature_vec_func(inputs[-1:, 2:4]), dim=0)

        assert feature_vec_a.shape == (self.Df, self.d), \
            "Feature vec a shape is " + str(feature_vec_a.shape) \
            + ". It should have shape ({}, {}).".format(self.Df, self.d)
        assert feature_vec_b.shape == (self.Df, self.d), \
            "Feature vec b shape is " + str(feature_vec_a.shape) \
            + ". It should have shape ({}, {}).".format(self.Df, self.d)

        # (1, Df)
        weights_a = self.weights_a[z].forward(inputs[-1:, 0:2])
        assert weights_a.shape == (1, self.Df)
        weights_b = self.weights_b[z].forward(inputs[-1:, 2:4])
        assert weights_b.shape == (1, self.Df)

        # (1, Df) * (Df, d) -> (1, d)
        out_a = torch.matmul(weights_a, feature_vec_a)
        assert out_a.shape == (1, self.d)
        out_b = torch.matmul(weights_b, feature_vec_b)
        assert out_b.shape == (1, self.d)

        out = torch.cat((out_a, out_b), dim=-1)
        out = torch.squeeze(out, dim=0)
        out = inputs[-1] + out
        assert out.shape == (self.D,)
        return out, (h_t, c_t)




