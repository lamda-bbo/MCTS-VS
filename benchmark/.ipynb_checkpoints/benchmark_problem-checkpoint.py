import numpy as np
from benchmark.synthetic_function import Hartmann, Levy


class FunctionBenchmark:
    def __init__(self, func, dims, valid_idx):
        assert func.dims == len(valid_idx)
        self.func = func
        self.dims = dims
        self.valid_idx = valid_idx
        self.lb = func.lb[0] * np.ones(dims)
        self.ub = func.ub[0] * np.ones(dims)
        self.opt_val = func.opt_val
        
    def __call__(self, x):
        assert len(x) == self.dims
        return self.func(x[self.valid_idx])
    
    
hartmann6 = Hartmann(6, True)
hartmann6_50 = FunctionBenchmark(hartmann6, 50, list(range(6)))
hartmann6_100 = FunctionBenchmark(hartmann6, 100, list(range(6)))

levy10 = Levy(10, True)
levy10_50 = FunctionBenchmark(levy10, 50, list(range(10)))
levy20 = Levy(20, True)
levy20_50 = FunctionBenchmark(levy20, 50, list(range(20)))
    
synthetic_function_problem = {
    'hartmann6': hartmann6,
    'hartmann6_50': hartmann6_50,
    'hartmann6_100': hartmann6_100,
    'levy10': levy10,
    'levy10_50': levy10_50,
    'levy20': levy20,
    'levy20_50': levy20_50,
}


if __name__ == '__main__':
    x = np.random.randn(50)
    print(x[: 6])
    print(synthetic_function_problem['hartmann6_50'](x))
