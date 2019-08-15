import torch

from ssm_ptc.transformations import BaseTransformation


class LinearTransformation(BaseTransformation):

    def __init__(self, K, d_in, d_out):
        super(LinearTransformation, self).__init__(d_out)

        self.As = torch.randn(K, d_in, d_out)

    @property
    def params(self):
        return [self.As]

    def transform(self, inputs):
        """

        :param inputs: (T, d_in)
        :return: (T, D)
        """

        K, d_in, d_out = self.As.shape

        T, _ = inputs.shape # (T, d_in)
        assert d_in.shape == inputs.shape[1]

        inputs = inputs.unsqueeze(0) # (1, T, d_in)

        out = torch.matmul(inputs, self.As)
        assert out.shape == (K, T, d_out)
        out = out.transpose(0,1)  # (T, K, D)

        return out

