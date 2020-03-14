import torch
import numpy as np

from ssm_ptc.transformations.base_transformation import BaseTransformation
from ssm_ptc.utils import check_and_convert_to_tensor

from project_ssms.unit_transformations.unit_direction_speedfree_transformation \
    import UnitDirectionSpeedFreeTransformation
from project_ssms.unit_transformations.unit_momentum_direction_transformation \
    import UnitMomentumDirectionTransformation
from project_ssms.unit_transformations.unit_direction_transformation import UnitDirectionTransformation
from project_ssms.unit_transformations.unit_direction_transformation_with_input \
    import UnitDirectionTransformationWithInput

UNIT_TRANSFORMATION_CLASSES = dict(
    momentum_direction=UnitMomentumDirectionTransformation,
    direction=UnitDirectionTransformation,
    direction_with_input=UnitDirectionTransformationWithInput,
    direction_speedfree=UnitDirectionSpeedFreeTransformation
)


class GridSingleTransformation(BaseTransformation):
    """
    Receive inputs in 4 dimension, but only perform transformation in the first 2 dimensions
    """
    def __init__(self, K, D, x_grids, y_grids, unit_transformation, **unit_transformation_kwargs):
        super(GridSingleTransformation, self).__init__(K, D)

        assert D == 2

        self.x_grids = x_grids  # (x_0, x_1, ..., x_m)
        self.y_grids = y_grids  # (y_0, y_1, ..., y_n)

        self.G = (len(self.x_grids) - 1) * (len(self.y_grids) - 1)

        unit_tran = UNIT_TRANSFORMATION_CLASSES.get(unit_transformation, None)
        if unit_tran is None:
            raise ValueError("Invalid unit transformation model: {}. Must be one of {}".
                             format(unit_transformation, list(UNIT_TRANSFORMATION_CLASSES.keys())))

        self.transformations_a = [unit_tran(K=self.K, D=self.D*2, **unit_transformation_kwargs)
                                  for _ in range(self.G)]

    def transform(self, inputs_self, masks_a=None, memory_kwargs_a=None):
        """
        Transform based on the current grid
        :param inputs_self: (T, 2)
        :param masks_a:
        :param memory_kwargs_a:
        :return: outputs (T, K, 4)
        """

        inputs_self = check_and_convert_to_tensor(inputs_self)

        if memory_kwargs_a is None:
            inputs_other = None
        else:
            inputs_other = memory_kwargs_a.get("inputs_other", None)

        T, _ = inputs_self.shape

        # perform transform on data
        # use the idea of mask

        if masks_a is None:
            masks_a = self.get_masks(inputs_self)

        memory_kwargs_a = memory_kwargs_a or {}

        output_a = 0
        for g in range(self.G):
            t_a = self.transformations_a[g].transform(inputs_self, inputs_other, **memory_kwargs_a)
            output_a = output_a + t_a * masks_a[g][:, None, None]

        assert output_a.shape == (T, self.K, 2)

        return output_a

    def transform_condition_on_z(self, z, inputs_self, memory_kwargs_a=None, inputs_other=None):
        """

        :param z: an integer
        :param inputs_self: (T_pre, 2)
        :param memory_kwargs_a:
        :param inputs_other: (T_pre, 2)
        :return: (D, )
        """
        memory_kwargs_a = memory_kwargs_a or {}

        # decide which transformation to use
        g_a = self.find_grid_index(inputs_self[-1], self.x_grids, self.y_grids)

        out_a = self.transformations_a[g_a].transform_condition_on_z(z, inputs_self, inputs_other, **memory_kwargs_a)

        assert out_a.shape == (2,)

        return out_a

    @staticmethod
    def find_grid_index(point, x_grids, y_grids):
        """

        :param point: (2, )
        :param x_grids: (G+1, )
        :param y_grids: (G+1, )
        :return: g: an integer in [0, G)
        """
        assert point.shape == (2,)
        g = 0
        find = False
        for i in range(len(x_grids) - 1):
            for j in range(len(y_grids) - 1):
                cond_x = x_grids[i] <= point[0] <= x_grids[i + 1]
                cond_y = y_grids[j] <= point[1] <= y_grids[j + 1]
                if (cond_x & cond_y):
                    find = True
                    break
                g = g + 1
            if find:
                break
        if not find:
            raise ValueError("value {} out of the grid world.".format(point.numpy()))
        return g

    def get_masks(self, data):
        """

        :param data: (T, 2)
        :return: two lists of masks, each list contains G masks, where each mask is a binary-valued array of length T
        """

        data = check_and_convert_to_tensor(data)
        masks_a = []
        for i in range(len(self.x_grids) - 1):
            for j in range(len(self.y_grids) - 1):
                if i == 0:
                    cond_x = (self.x_grids[i] <= data[:, 0]) & (data[:, 0] <= self.x_grids[i + 1])
                else:
                    cond_x = (self.x_grids[i] < data[:, 0]) & (data[:, 0] <= self.x_grids[i + 1])
                if j == 0:
                    cond_y = (self.y_grids[j] <= data[:, 1]) & (data[:, 1] <= self.y_grids[j + 1])
                else:
                    cond_y = (self.y_grids[j] < data[:, 1]) & (data[:, 1] <= self.y_grids[j + 1])

                mask = (cond_x & cond_y).double()
                masks_a.append(mask)

        masks_a = torch.stack(masks_a, dim=0)
        assert torch.all(masks_a.sum(dim=0) == 1)

        return masks_a



