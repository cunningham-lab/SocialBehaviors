import torch
import numpy as np

from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.utils import check_and_convert_to_tensor

from project_ssms.single_transformations.base_single_transformation import BaseSingleTransformation
from project_ssms.single_transformations.single_direction_transformation import SingleDirectionTransformation


class GridTransformation(BaseTransformation):
    def __init__(self, K, D, x_grids, y_grids, Df=None, feature_vec_func=None,
                 lags=2, momentum_weights=None, acc_factor=2):
        super(GridTransformation, self).__init__(K, D)
        self.d = int(self.D / 2)

        self.x_grids = x_grids  # (x_0, x_1, ..., x_m)
        self.y_grids = y_grids  # (y_0, y_1, ..., y_n)
        self.Df = Df
        self.feature_vec_func = feature_vec_func
        self.momentum_lags = lags
        self.momentum_weights = momentum_weights
        self.acc_factor = acc_factor

        self.G = (len(self.x_grids) - 1) * (len(self.y_grids) - 1)

        self.transformations_a = [SingleDirectionTransformation(K=self.K, D=self.D, Df=self.Df,
                                                                momentum_lags=self.momentum_lags,
                                                                momentum_weights=self.momentum_weights,
                                                                feature_vec_func=self.feature_vec_func,
                                                                acc_factor=self.acc_factor) for _ in range(self.G)]

        self.transformations_b = [SingleDirectionTransformation(K=self.K, D=self.D, Df=self.Df,
                                                                momentum_lags=self.momentum_lags,
                                                                momentum_weights=self.momentum_weights,
                                                                feature_vec_func=self.feature_vec_func,
                                                                acc_factor=self.acc_factor) for _ in range(self.G)]

    @property
    def params(self):
        params = ()
        for g in range(self.G):
            params = params + self.transformations_a[g].params
            params = params + self.transformations_b[g].params
        return params

    @params.setter
    def params(self, values):
        i = 0
        for g in range(self.G):
            self.transformations_a[g].params = values[i]
            i = i + 1
            self.transformations_b[g].params = values[i]
            i = i + 1

    def permute(self, perm):
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

        """
        # a map: inputs -> grids, (T, 2), should be index (integer-valued)

        # get batches data, a list of length G
        grid_data_a = []
        grid_data_b = []
        # get time index, a list of length G
        grid_time_a = []  #  gird_time[0] is an array of time index when the point is in grid 0
        grid_time_b = []

        # transform in batch
        for g in range(G):
            x_a = T[0][g](grid_data_a[g])  # during grid_time_a
            x_b = T[1][g](grid_data_b[g])  # during grid_time_b
            # make it back to thhe grid
            
        """
        inputs = check_and_convert_to_tensor(inputs)
        T, _ = inputs.shape

        # perform transform on data
        # use the idea of mask

        if masks is None:
            masks_a, masks_b = self.get_masks(inputs)
        else:
            assert isinstance(masks, tuple)
            masks_a, masks_b = masks

        memory_kwargs_a = memory_kwargs_a or {}
        memory_kwargs_b = memory_kwargs_b or {}

        output_a = 0
        output_b = 0
        for g in range(self.G):
            t_a = self.transformations_a[g].transform(inputs[:, 0:2], inputs[:, 2:4], **memory_kwargs_a)
            output_a = output_a + t_a * masks_a[g][:, None, None]

            t_b = self.transformations_b[g].transform(inputs[:, 2:4], inputs[:, 0:2], **memory_kwargs_b)
            output_b = output_b + t_b * masks_b[g][:, None, None]

        assert output_a.shape == (T, self.K, self.d)
        assert output_b.shape == (T, self.K, self.d)

        outputs = torch.cat((output_a, output_b), dim=-1)
        assert outputs.shape == (T, self.K, self.D)

        return outputs

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

        # decide which transformation ot use
        g_a = self.find_grid_index(inputs[-1, :2], self.x_grids, self.y_grids, self.G)
        g_b = self.find_grid_index(inputs[-1, 2:], self.x_grids, self.y_grids, self.G)

        out_a = self.transformations_a[g_a].transform_condition_on_z(z, inputs[:, :2], inputs[:, 2:], **memory_kwargs_a)
        out_b = self.transformations_b[g_b].transform_condition_on_z(z, inputs[:, 2:], inputs[:, :2], **memory_kwargs_b)

        assert out_a.shape == (2, )
        assert out_b.shape == (2, )

        out = torch.cat((out_a, out_b), dim=0)
        assert out.shape == (self.D, )
        return out

    @staticmethod
    def find_grid_index(point, x_grids, y_grids, G):
        """

        :param point: (2, )
        :param x_grids: (G+1, )
        :param y_grids: (G+1, )
        :return: g: an integer in [0, G)
        """
        assert point.shape == (2, )
        g = 0
        for i in range(len(x_grids)-1):
            for j in range(len(y_grids)-1):
                cond_x = x_grids[i] < point[0] <= x_grids[i+1]
                cond_y = y_grids[j] < point[1] <= y_grids[j+1]
                if (cond_x & cond_y):
                    break
                g = g + 1
        if g == G:
            raise ValueError("value {} out of the grid world.".format(point.numpy()))
        return g

    def get_masks(self, data):
        """

        :param data: (T, 4)
        :return: two lists of masks, each list contains G masks, where each mask is a binary-valued array of length T
        """

        data = check_and_convert_to_tensor(data)
        masks_a = []
        masks_b = []
        for i in range(len(self.x_grids)-1):
            for j in range(len(self.y_grids)-1):
                cond_x = (self.x_grids[i] < data[:, 0]) & (data[:, 0] <= self.x_grids[i + 1])
                cond_y = (self.y_grids[j] < data[:, 1]) & (data[:, 1] <= self.y_grids[j + 1])
                mask = (cond_x & cond_y).double()
                masks_a.append(mask)

                cond_x = (self.x_grids[i] < data[:, 2]) & (data[:, 2] <= self.x_grids[i + 1])
                cond_y = (self.y_grids[j] < data[:, 3]) & (data[:, 3] <= self.y_grids[j + 1])
                mask = (cond_x & cond_y).double()
                masks_b.append(mask)

        return masks_a, masks_b



