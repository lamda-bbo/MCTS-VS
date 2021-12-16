import torch
import botorch
import numpy as np
import pandas as pd
import argparse
import random
from benchmark import synthetic_function_problem
from utils import save_results


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=1000, type=int)
parser.add_argument('--root_dir', default='simple_logs', type=str)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)
f = synthetic_function_problem[args.func]


dims = f.dims
valid_dims = len(f.valid_idx)


def evaluation_function(parameterization):
    x = np.array([parameterization['x'+str(i)] for i in range(dims)])
    return {'objective': (f(x), 0.0)}


parameters = [
    {'name': f'x{i}', 'type': 'range', 'bounds': [f.lb[i], f.ub[i]], 'value_type': 'float'} for i in range(dims)
]


from ax.modelbridge.strategies.alebo import ALEBOStrategy

alebo_strategy = ALEBOStrategy(D=dims, d=valid_dims, init_size=5)

from ax.service.managed_loop import optimize

best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    experiment_name=args.func,
    objective_name='objective',
    evaluation_function=evaluation_function,
    minimize=False,
    total_trials=10,
    generation_strategy=alebo_strategy
)