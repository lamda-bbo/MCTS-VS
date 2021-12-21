import numpy as np
import re
from benchmark.synthetic_function import Ackley, Branin, Hartmann, Levy, Rosenbrock
from benchmark.tracker import Tracker
from benchmark.nas_benchmark import NasBench
from benchmark.rover_function import Rover


class FunctionBenchmark:
    def __init__(self, func, dims, valid_idx, save_config):
        assert func.dims == len(valid_idx)
        self.func = func
        self.dims = dims
        self.valid_idx = valid_idx
        self.lb = func.lb[0] * np.ones(dims)
        self.ub = func.ub[0] * np.ones(dims)
        self.opt_val = func.opt_val
        
        self.save_config = save_config
        self.tracker = Tracker(
            self.save_config['save_interval'],
            self.save_config
        )
        
    def __call__(self, x):
        assert len(x) == self.dims
        result = self.func(x[self.valid_idx])
        self.tracker.track(result)
        return result
    

ackley20 = Ackley(20, True)
branin2 = Branin(2, True)
hartmann6 = Hartmann(6, True)
levy10 = Levy(10, True)
levy20 = Levy(20, True)
rosenbrock20 = Rosenbrock(20, True)


def get_problem(func_name, save_config):
    """
    save_config: {'save_interval': int, 'root_dir': str, 'algo': str, 'func': str, 'seed': int}
    """
    if func_name in ['nasbench', 'rover']:
        if func_name == 'nasbench':
            return FunctionBenchmark(NasBench(), 36, list(range(36)), save_config)
        if func_name == 'rover':
            return FunctionBenchmark(Rover(), 60, list(range(60)), save_config)
        else:
            assert 0
    else:
        split_result = func_name.split('_')

        if len(split_result) == 1:
            func = split_result[0]
            dims = None
        else:
            func, dims = split_result
            dims = int(dims)

        valid_dims = int(re.findall(r'\d+', func)[0])
        dims = valid_dims if dims is None else dims
        return FunctionBenchmark(eval(func), dims, list(range(valid_dims)), save_config)


if __name__ == '__main__':
    x = np.random.randn(50)
    save_config = {
        'save_interval': 3,
        'root_dir': 'logs',
        'algo': 'bo',
        'func': 'lecy10_50',
        'seed': 42
    }
    func = get_problem('levy10_50', save_config)
    
    for _ in range(10):
        func(x)
