from abc import ABCMeta
import numpy as np


class SyntheticFunction(metaclass=ABCMeta):
    def __init__(self, dims, negate, lb, ub, opt_val, opt_point):
        self.dims = dims
        self.negate = negate
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.opt_val = opt_val
        self.opt_point = np.array(opt_point)
        
    def __call__(self, x):
        pass

    
class Ackley(SyntheticFunction):
    def __init__(self, dims=20, negate=False):
        SyntheticFunction.__init__(
            self, 
            dims, 
            negate, 
            -10 * np.ones(dims),
            10 * np.ones(dims),
            0,
            np.array([0]*dims)
        )
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
        if self.negate:
            result = - result
        return result
    
    
class Branin(SyntheticFunction):
    def __init__(self, dims=2, negate=False):
        assert dims == 2
        SyntheticFunction.__init__(
            self,
            dims, 
            negate,
            np.array([-5, -5]),
            np.array([15, 15]),
            -0.397887,
            np.array([-np.pi, 12.275]), # [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]
        )
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        t1 = x[1] \
            - 5.1 / (4*np.pi**2) * x[0]**2 \
            + 5 / np.pi * x[0] - 6
        t2 = 10 * (1 - 1/(8*np.pi)) * np.cos(x[0])
        result = t1**2 + t2 + 10
        if self.negate:
            result = - result
        return result
    
    
class Hartmann(SyntheticFunction):
    def __init__(self, dims=6, negate=False):
        assert dims == 6
        SyntheticFunction.__init__(
            self, 
            dims, 
            negate, 
            0 * np.ones(dims),
            1 * np.ones(dims),
            -3.32237,
            np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        )
        
        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self.A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ])
        self.P = np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ])
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        inner_sum = np.sum(self.A * (x.reshape(1, -1) - 0.0001 * self.P) ** 2, axis=-1)
        result = - np.sum(self.alpha * np.exp(-inner_sum), axis=-1)
        if self.negate:
            result = - result
        return result
    
    
class HartmannExtend(SyntheticFunction):
    def __init__(self, dims=30, negate=False):
        assert dims % 6 == 0
        SyntheticFunction.__init__(
            self, 
            dims, 
            negate, 
            0 * np.ones(dims),
            1 * np.ones(dims),
            -3.32237,
            np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        )
        
        self.func = Hartmann(6, negate)
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        result = 0
        for i in range(int(self.dims / 6)):
            result += self.func(x[i*6: (i+1)*6])
        return result
    
    
class Levy(SyntheticFunction):
    def __init__(self, dims=10, negate=False):
        SyntheticFunction.__init__(
            self, 
            dims, 
            negate, 
            -10 * np.ones(dims),
            10 * np.ones(dims),
            0,
            np.ones(dims)
        )
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        w = 1 + (x - 1.0) / 4.0
        result = np.sin(np.pi * w[0]) ** 2 + \
            np.sum((w[1:self.dims - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dims - 1] + 1) ** 2)) + \
            (w[self.dims - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dims - 1])**2)
        if self.negate:
            result = - result
        return result
    
    
class Rosenbrock(SyntheticFunction):
    def __init__(self, dims=2, negate=False):
        SyntheticFunction.__init__(
            self, 
            dims, 
            negate, 
            -5 * np.ones(dims),
            10 * np.ones(dims),
            0,
            np.ones(dims)
        )
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = np.sum(100 * (x[1: ] - x[: -1]**2)**2 + (x[: -1] - 1)**2)
        if self.negate:
            result = - result
        return result
    
    
if __name__ == '__main__':
    # x = np.random.randn(6)
    # func = Hartmann(6, True)
    # print(func(x))
    # import torch
    # from botorch.test_functions import Hartmann, Branin, Levy
    # func = Hartmann(6, negate=True)
    # print(func(torch.tensor(x)))
    
    func = Branin(2, True)
    print(func(np.array([-np.pi, 12.275])))
    print(func(np.array([np.pi, 2.275])))
    print(func(np.array([9.42478, 2.475])))
    
    func = Ackley(10, True)
    print(func(np.zeros(10)))
    
    func = Rosenbrock(10, True)
    print(func(np.ones(10)))