import numpy as np


# openai: https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat:
    def __init__(self, shape):
        self.n = 0
        self.m = np.zeros(shape)
        self.s = np.zeros(shape)

    def push(self, x):
        x = np.array(x)
        assert x.shape == self.m.shape
        self.n += 1
        if self.n == 1:
            self.m = x
        else:
            old_m = self.m.copy()
            self.m = old_m + (x - old_m) / self.n
            self.s = self.s + (x - old_m) * (x - self.m)

    @property
    def mean(self):
        return self.m

    @property
    def var(self):
        return self.s / (self.n - 1) if self.n > 1 else np.zeros(self.m.shape)

    @property
    def std(self):
        return np.sqrt(self.var)