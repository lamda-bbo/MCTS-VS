import numpy as np
import pandas as pd
import re
import time
from benchmark.synthetic_function import Ackley, Branin, Hartmann, Levy, Rosenbrock, HartmannExtend
from benchmark.tracker import Tracker, save_results
from benchmark.rover_function import Rover


class FunctionBenchmark:
    def __init__(self, func, dims, valid_idx, save_config):
        assert func.dims == len(valid_idx)
        self.func = func
        self.dims = dims
        self.valid_idx = valid_idx
        self.lb = func.lb[0] * np.ones(dims)
        self.ub = func.ub[0] * np.ones(dims)
        
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
    
    
class RLBenchmark:
    def __init__(self, func, dims, valid_idx, save_config):
        assert func.dims == len(valid_idx)
        self.func = func
        self.dims = dims
        self.valid_idx = valid_idx
        self.lb = func.lb[0] * np.ones(dims)
        self.ub = func.ub[0] * np.ones(dims)
        self.save_config = save_config
        
        self.counter = 0
        self.n_samples = 0
        self.start_time = time.time()
        self.curt_best = float('-inf')
        self.best_value_trace = []
        
    def __call__(self, x):
        assert len(x) == self.dims
        result, n_samples = self.func(x[self.valid_idx])
        self.track(result, n_samples)
        return result
    
    def track(self, result, n_samples):
        self.counter += 1
        self.n_samples += n_samples
        if result > self.curt_best:
            self.curt_best = result
        self.best_value_trace.append((
            self.counter,
            self.curt_best,
            time.time() - self.start_time,
            self.n_samples
        ))
        
        if self.counter % 50 == 0:
            df_data = pd.DataFrame(self.best_value_trace, columns=['x', 'y', 't', 'n_samples'])
            save_results(
                self.save_config['root_dir'],
                self.save_config['algo'],
                self.save_config['func'],
                self.save_config['seed'],
                df_data,
            )
    

hartmann6 = Hartmann(6, True)
levy10 = Levy(10, True)


def get_problem(func_name, save_config, seed=2021):
    """
    save_config: {'save_interval': int, 'root_dir': str, 'algo': str, 'func': str, 'seed': int}
    """
    if func_name in ['nasbench', 'rover', 'HalfCheetah', 'Walker2d', 'Hopper']:
        if func_name in ['HalfCheetah', 'Walker2d', 'Hopper']:
            from benchmark.rl_benchmark import RLEnv
            
        if func_name == 'nasbench':
            from benchmark.nas_benchmark import NasBench
            return FunctionBenchmark(NasBench(seed=seed), 36, list(range(36)), save_config)
        elif func_name == 'rover':
            return FunctionBenchmark(Rover(), 60, list(range(60)), save_config)
        elif func_name == 'HalfCheetah' or func_name == 'Walker2d':
            return RLBenchmark(RLEnv(func_name+'-v2', seed), 102, list(range(102)), save_config)
        elif func_name == 'Hopper':
            return RLBenchmark(RLEnv('Hopper-v2', seed), 33, list(range(33)), save_config)
        else:
            assert 0
    elif func_name != 'nasbench' and func_name.startswith('nasbench'):
        if func_name == 'nasbench1shot1':
            from benchmark.hpo_benchmark import HpoNasBench
            dims = 33
            return FunctionBenchmark(HpoNasBench(name=func_name, seed=seed), dims, list(range(dims)), save_config)
        elif func_name == 'nasbench201':
            from benchmark.naslib_benchmark import NASLibBench
            dims = 30
            return FunctionBenchmark(NASLibBench(name=func_name, seed=seed), dims, list(range(dims)), save_config)
        elif func_name == 'nasbenchtrans':
            from benchmark.naslib_benchmark import NASLibBench
            dims = 24
            return FunctionBenchmark(NASLibBench(name='transbench101_micro', seed=seed), dims, list(range(dims)), save_config)
        elif func_name == 'nasbenchasr':
            from benchmark.naslib_benchmark import NASLibBench
            dims = 30
            return FunctionBenchmark(NASLibBench(name='asr', seed=seed), dims, list(range(dims)), save_config)
        else:
            assert 0
    else:
        split_result = func_name.split('_')

        if len(split_result) == 1:
            func = split_result[0]
            dims = None
        elif len(split_result) == 2:
            func, dims = split_result
            dims = int(dims)
        else:
            assert 0

        valid_dims = int(re.findall(r'\d+', func)[0])
        dims = valid_dims if dims is None else dims
        if func_name.startswith('hartmann') and split_result[0] != 'hartmann6':
            d = split_result[0].strip('hartmann')
            valid_dims = int(d)
            dims = int(func_name.split('_')[-1])
            return FunctionBenchmark(HartmannExtend(valid_dims, True), dims, list(range(valid_dims)), save_config)
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
