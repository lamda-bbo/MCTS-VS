import torch
import botorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
from benchmark import get_problem
from MCTSVS.MCTS import MCTS
from utils import save_args


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_300', type=str, 
                    choices=['hartmann6_300', 'hartmann6_500', 'levy10_100', 'levy10_300', 'nasbench', 'nasbench201', 'nasbench1shot1', 'nasbenchtrans', 'nasbenchasr', 'Hopper', 'Walker2d'])
parser.add_argument('--max_samples', default=600, type=int)
parser.add_argument('--feature_batch_size', default=2, type=int)
parser.add_argument('--sample_batch_size', default=3, type=int)
parser.add_argument('--min_num_variables', default=3, type=int)
parser.add_argument('--select_right_threshold', default=5, type=int)
parser.add_argument('--turbo_max_evals', default=50, type=int)
parser.add_argument('--k', default=20, type=int)
parser.add_argument('--Cp', default=0.1, type=float)
parser.add_argument('--ipt_solver', default='bo', type=str)
parser.add_argument('--uipt_solver', default='bestk', type=str)
parser.add_argument('--root_dir', default='synthetic_logs', type=str)
parser.add_argument('--dir_name', default=None, type=str)
parser.add_argument('--postfix', default=None, type=str)
parser.add_argument('--seed', default=2021, type=int)
args = parser.parse_args()

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)

algo_name = 'mcts_vs_{}'.format(args.ipt_solver)
if args.postfix is not None:
    algo_name += ('_' + args.postfix)
save_config = {
    'save_interval': 50,
    'root_dir': 'logs/' + args.root_dir,
    'algo': algo_name,
    'func': args.func if args.dir_name is None else args.dir_name,
    'seed': args.seed
}
f = get_problem(args.func, save_config, args.seed)

save_args(
    'config/' + args.root_dir,
    algo_name,
    args.func if args.dir_name is None else args.dir_name,
    args.seed,
    args
)

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
    k=args.k,
    split_type='mean',
    ipt_solver=args.ipt_solver, 
    uipt_solver=args.uipt_solver,
    turbo_max_evals=args.turbo_max_evals,
)

agent.search(max_samples=args.max_samples, verbose=False)

print('best f(x):', agent.value_trace[-1][1])
