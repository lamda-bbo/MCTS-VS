import torch
import botorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
from benchmark import get_problem
from LamctsVS_theory.MCTS import MCTS
from utils import save_args


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=200, type=int)
parser.add_argument('--feature_batch_size', default=2, type=int)
parser.add_argument('--sample_batch_size', default=3, type=int)
parser.add_argument('--min_num_variables', default=8, type=int)
parser.add_argument('--select_right_threshold', default=5, type=int)
parser.add_argument('--turbo_max_evals', default=50, type=int)
parser.add_argument('--Cp', default=0, type=float)
parser.add_argument('--ipt_solver', default='bo', type=str)
parser.add_argument('--uipt_solver', default='bestk', type=str)
parser.add_argument('--root_dir', default='theory_logs', type=str)
parser.add_argument('--postfix', default=None, type=str)
parser.add_argument('--seed', default=2021, type=int)
args = parser.parse_args()

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)

algo_name = 'lamcts_vs_theory'
if args.postfix is not None:
    algo_name += ('_' + args.postfix)
save_config = {
    'save_interval': 50,
    'root_dir': 'logs/' + args.root_dir,
    'algo': algo_name,
    'func': args.func,
    'seed': args.seed
}
f = get_problem(args.func, save_config, args.seed)

save_args(
    'config/' + args.root_dir,
    algo_name,
    args.func,
    args.seed,
    args
)

# point = np.random.uniform(0, 1, f.dims)
# # eps = np.random.randn(f.dims) * 0.01
# # eps_point = np.clip(point + eps, f.lb, f.ub)
# eps = np.random.randn() * 0.01
# basis = np.zeros(f.dims)
# basis[3] = 1
# eps_point = np.clip(point + eps*basis, f.lb, f.ub)

# print(f(point))
# print(f(point) - f(eps_point))






agent = MCTS(
    func=f,
    dims=f.dims,
    lb=f.lb,
    ub=f.ub,
    feature_batch_size=args.feature_batch_size,
    sample_batch_size=args.sample_batch_size,
    Cp=args.Cp,
    min_num_variables=args.min_num_variables, 
    select_right_threshold=args.select_right_threshold, 
    split_type='mean',
    ipt_solver=args.ipt_solver, 
    uipt_solver=args.uipt_solver,
    turbo_max_evals=args.turbo_max_evals,
)

agent.search(max_samples=args.max_samples, verbose=False)

print(agent.selected_variables)

delta = []
for selected in agent.selected_variables:
    # sum_selected = selected.sum(axis=0)
    delta.append(len(set(range(len(f.valid_idx))) - set(selected)))
    
print(delta)
res = [np.sum(delta[: idx+1]) / (idx + 1) for idx in range(len(delta))]
print(res)
plt.plot(res)
plt.savefig('theory.png')

print('best f(x):', agent.value_trace[-1][1])
