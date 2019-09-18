import torch
import numpy as np

from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.utils import check_and_convert_to_tensor


class GridTransformation(BaseTransformation):
    """
    Learnable parameters: weights of each grid vertex
    weights of any random point = a weighted combination of weights of all the grid vertices
    """
    def __init__(self, K, D, x_grids, y_grids):
        super(GridTransformation, self).__init__(K, D)
        self.d = int(self.D / 2)

        self.x_grids = x_grids  # (x_0, x_1, ..., x_m)
        self.y_grids = y_grids  # (y_0, y_1, ..., y_n)

        # number of grids
        self.G = (len(self.x_grids) - 1) * (len(self.y_grids) - 1)

        # number of basis grid points
        self.GP = len(self.x_grids) * len(self.y_grids)

        # specify the weights of the grid points

    @property
    def params(self):
        # use weights
        params = ()
        for g in range(self.G):
            params = params + self.transformations_a[g].params
            params = params + self.transformations_b[g].params
        return params

    @params.setter
    def params(self, values):
        # set weights
        i = 0
        for g in range(self.G):
            self.transformations_a[g].params = values[i]
            i = i + 1
            self.transformations_b[g].params = values[i]
            i = i + 1

    def permute(self, perm):
        # permute weights
        for g in range(self.G):
            self.transformations_a[g].permute(perm)
            self.transformations_b[g].permute(perm)

    def transform(self, inputs, masks=None, memory_kwargs_a=None, memory_kwargs_b=None):
        """
        Transform based on the current grid
        :param inputs: (T, 4)
        :param masks:
        :param memory_kwargs_a:
        :param memory_kwargs_b:
        :return: outputs (T, K, 4)
        """

        # TODO: apply a softmax function for the weights for weighted combination of weights of the grid points

        pass

    def transform_condition_on_z(self, z, inputs, memory_kwargs_a=None, memory_kwargs_b=None):
        """

        :param z: an integer
        :param inputs: (T_pre, D)
        :param memory_kwargs_a:
        :param memory_kwargs_b:
        :return: (D, )
        """
        pass





