import torch
import numpy as np

from ssm_ptc.transformations.base_transformation import BaseTransformation


class LinearGridTransformation(BaseTransformation):
    def __init__(self, K, D, x_grids, y_grids):
        super(LinearGridTransformation, self).__init__(K, D)

        self.d = int(self.D / 2)

        self.x_grids = x_grids  # (x_0, x_1, ..., x_m)
        self.y_grids = y_grids  # (y_0, y_1, ..., y_n)

        # number of grids
        self.G = (len(self.x_grids) - 1) * (len(self.y_grids) - 1)

        # number of basis grid points
        self.GP = len(self.x_grids) * len(self.y_grids)

        # TODO: only need to specify weights, and make transformation (make it a function), that takes in weights and positions

        # basis transformations
        self.transformations_a = [unit_tran(K=self.K, D=self.D, **unit_transformation_kwargs)
                                  for _ in range(self.GP)]
        self.transformations_b = [unit_tran(K=self.K, D=self.D, **unit_transformation_kwargs)
                                  for _ in range(self.GP)]

    @property
    def params(self):
        params = ()
        for grid_point in range(self.GP):
            params = params + self.transformations_a[grid_point].params
            params = params + self.transformations_b[grid_point].params
        return params

    @params.setter
    def params(self, values):
        i = 0
        for grid_point in range(self.GP):
            self.transformations_a[grid_point].params = values[i]
            i = i + 1
            self.transformations_b[grid_point].params = values[i]
            i = i + 1

    def permute(self, perm):
        for grid_point in range(self.GP):
            self.transformations_a[grid_point].permute(perm)
            self.transformations_b[grid_point].permute(perm)

    def transform(self, inputs, masks=None, memory_kwargs_a=None, memory_kwargs_b=None):
        # TODO: do a 2D interpolation of weights, and use that for transformation

        # first, find the grid, so we can assign the four basis grid points

        # then compute the weights

        # finally, compute the transformation based on weights and positions (directions)

        pass