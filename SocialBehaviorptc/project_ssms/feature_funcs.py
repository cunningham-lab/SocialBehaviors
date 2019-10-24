import torch
import numpy as np

from project_ssms.constants import ARENA_XMIN, ARENA_XMAX, ARENA_YMIN, ARENA_YMAX

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# some constants

# specify some locations
WATER = torch.tensor([50, 50], dtype=torch.float64)
FOOD = torch.tensor([270, 50], dtype=torch.float64)
NEST = torch.tensor([270, 330], dtype=torch.float64)
CORNER = torch.tensor([50, 330], dtype=torch.float64)

arena_xmax = 320
arena_ymax = 370


def unit_vector_vec_np(vectorvec):
    return vectorvec / np.linalg.norm(vectorvec, axis=1, keepdims=True)


def unit_vector_vec(vectorvec):
    return vectorvec / torch.norm(vectorvec, dim=-1, keepdim=True)


def angle_between_vec(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            angle_between((1, 0, 0), (1, 0, 0))
            0.0
            angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector_vec(v1)
    v2_u = unit_vector_vec(v2)
    dotted = np.einsum('ik,ik->i', v1_u, v2_u)
    return np.arccos(np.clip(dotted, -1.0, 1.0))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def unit_vector_to_other_np(self, other):
    diff = other - self
    return unit_vector_vec_np(diff)  # (T, 2)


def unit_vector_to_other(self, other):
    return unit_vector_vec(self-other)


def unit_vector_to_fixed_loc_np(self, fixed_loc):
    to_fixed_loc = fixed_loc[None, ] - self
    return unit_vector_vec_np(to_fixed_loc)  # (T,2)


def unit_vector_to_fixed_loc(self, fixed_loc):
    to_fixed_loc = fixed_loc[None, ] - self
    return unit_vector_vec(to_fixed_loc)  # (T,2)


def unit_center_to_head(self, other):
    assert len(self.shape) == 2
    left_ear = self[:, 2:4]  # shape (T,2)
    right_ear = self[:, 4:6]  # shape (T,2)
    head_center = (left_ear + right_ear) / 2

    center = self[:, 6:8]
    return unit_vector_vec(head_center - center)  # (T,2)


def gazing_angle(self, other):
    head_center = (self[:, 2:4] + self[:, 4:6]) / 2
    head_center_to_tip = self[:, 0:2] - head_center
    head_center_to_other_tip = other[:, 0:2] - head_center
    angle = angle_between_vec(head_center_to_tip, head_center_to_other_tip)
    return angle[:, None]  # (T,1)


def feature_func_single(s, o):
    """
    :param s: self, (T, 2)
    :param o: other, (T, 2)
    :return: features, (T, 2 * Df)
    """

    features_0 = [unit_vector_to_other(s, o)]
    features_rest = [unit_vector_to_fixed_loc(s, pos) for pos in [WATER, NEST, FOOD, CORNER]]
    features = features_0 + features_rest
    features = torch.cat(features, dim=-1)
    return features


def feature_vec_func(s, o):
    """
    :param s: self, (T, 2)
    :param o: other, (T, 2)
    :return: features, (T, Df, 2)
    """
    features_0 = [unit_vector_to_other(s, o)]
    features_rest = [unit_vector_to_fixed_loc(s, pos) for pos in [WATER, NEST, FOOD, CORNER]]
    features = features_0 + features_rest # each is a tensor of shape (T,2), and there are Df items of them
    features = torch.stack(features, dim=1)
    return features


def feature_direction_vec(s, corners):
    """

    :param s: self, (T, 2)
    :param corners: a list or array of 4, each is (2,)
    :return: (T, 4, 2) unit vecs to each corner
    """
    corners = torch.tensor(corners, dtype=torch.float64, device=s.device)
    features = [unit_vector_to_fixed_loc(s, corners[i]) for i in range(4)]  # each is a tensor of shape (T,2), and there are Df items of them
    features = torch.stack(features, dim=1)
    return features


# feature_funcs
#CORNERS = torch.tensor([[ARENA_XMIN, ARENA_YMIN], [ARENA_XMIN, ARENA_YMAX],
 #                       [ARENA_XMAX, ARENA_YMIN], [ARENA_XMAX, ARENA_YMAX]], dtype=torch.float64)
CORNERS = np.array([[ARENA_XMIN, ARENA_YMIN], [ARENA_XMIN, ARENA_YMAX],
                    [ARENA_XMAX, ARENA_YMIN], [ARENA_XMAX, ARENA_YMAX]])


def f_corner_vec_func(s):
    return feature_direction_vec(s, CORNERS)
