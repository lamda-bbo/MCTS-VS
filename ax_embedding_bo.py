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
from benchmark import synthetic_function_problem
from utils import save_results


def evaluation_function(parameterization):
    x = np.array([parameterization['x'+str(i)] for i in range(dims)])
    return {'objective': (f(x), 0.0)}


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=1000, type=int)
parser.add_argument('--strategy', default='rembo', type=str)
parser.add_argument('--root_dir', default='simple_logs', type=str)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
# ax.models.random.sobol.SobolGenerator.seed(args.seed)

f = synthetic_function_problem[args.func]
dims = f.dims
valid_dims = len(f.valid_idx)

parameters = [
    {'name': f'x{i}', 'type': 'range', 'bounds': [f.lb[i], f.ub[i]], 'value_type': 'float'} for i in range(dims)
]

if args.strategy == 'rembo':
    embedding_strategy = REMBOStrategy(D=dims, d=valid_dims, init_per_proj=2)
elif args.strategy == 'hesbo':
    embedding_strategy = HeSBOStrategy(D=dims, d=valid_dims, init_per_proj=2)
elif args.strategy == 'alebo':
    embedding_strategy = ALEBOStrategy(D=dims, d=valid_dims, init_size=10)
else:
    assert 0, 'Strategy should be rembo, hesbo, alebo'

# TODO: fix the seed
    
best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    experiment_name=args.func,
    objective_name='objective',
    evaluation_function=evaluation_function,
    minimize=False,
    total_trials=10,
    generation_strategy=embedding_strategy,
    random_seed=args.seed,
)

objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])

best_value = np.maximum.accumulate(objectives)

print(best_value[-1])

plt.plot(range(len(best_value)), best_value)
plt.savefig('./tmp.png')