import torch
import botorch
import ax
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import REMBOStrategy, HeSBOStrategy
from ax.service.managed_loop import optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
from benchmark import get_problem
from utils import save_results, save_args


def evaluation_function(parameterization):
    x = np.array([parameterization['x'+str(i)] for i in range(dims)])
    return {'objective': (func(x), 0.0)}


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=600, type=int)
parser.add_argument('--active_dims', default=6, type=int)
parser.add_argument('--strategy', default='rembo', type=str)
parser.add_argument('--root_dir', default='synthetic_logs', type=str)
parser.add_argument('--seed', default=2021, type=int)
args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)

save_config = {
    'save_interval': 50,
    'root_dir': 'logs/' + args.root_dir,
    'algo': args.strategy,
    'func': args.func,
    'seed': args.seed
}
func = get_problem(args.func, save_config)
dims = func.dims

save_args(
    'config/' + args.root_dir,
    args.strategy,
    args.func,
    args.seed,
    args
)

parameters = [
    {'name': f'x{i}', 'type': 'range', 'bounds': [func.lb[i], func.ub[i]], 'value_type': 'float'} for i in range(dims)
]

if args.strategy == 'rembo':
    embedding_strategy = REMBOStrategy(D=dims, d=args.active_dims, init_per_proj=2)
elif args.strategy == 'hesbo':
    embedding_strategy = HeSBOStrategy(D=dims, d=args.active_dims, init_per_proj=2)
elif args.strategy == 'alebo':
    embedding_strategy = ALEBOStrategy(D=dims, d=args.active_dims, init_size=10)
else:
    assert 0, 'Strategy should be rembo, hesbo, alebo'

best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    experiment_name=args.func,
    objective_name='objective',
    evaluation_function=evaluation_function,
    minimize=False,
    total_trials=args.max_samples,
    generation_strategy=embedding_strategy,
    random_seed=args.seed,
)

print('best f(x):', func.tracker.best_value_trace[-1])
