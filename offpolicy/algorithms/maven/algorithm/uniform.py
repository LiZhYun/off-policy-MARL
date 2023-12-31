import torch as th


class Uniform:

    def __init__(self, args):
        self.args = args
        self.noise_distrib = th.distributions.one_hot_categorical.OneHotCategorical(th.tensor([1/self.args.noise_dim for _ in range(self.args.noise_dim)]))

    def sample(self, num):
        return self.noise_distrib.sample(num)

    def update_returns(self, state, noise, returns, test_mode, t):
        pass