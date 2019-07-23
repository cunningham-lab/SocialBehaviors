

class BaseTransition():

    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, values):
        raise NotImplementedError

    @property
    def transition_matrix(self, data, input):
        raise NotImplementedError

    def permute(self, perm):
        raise NotImplementedError