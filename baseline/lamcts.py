# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import torch
import botorch
import numpy as np
import pandas as pd
import random
import argparse
from inner_optimizer import MCTS
from benchmark import get_problem
from utils import save_results, save_args


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=600, type=int)
parser.add_argument('--Cp', default=0.1, type=float)
parser.add_argument('--solver_type', default='bo', type=str)
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
    'algo': 'lamcts_{}'.format(args.solver_type),
    'func': args.func,
    'seed': args.seed
}
f = get_problem(args.func, save_config, args.seed)

save_args(
    'config/' + args.root_dir,
    'lamcts_{}'.format(args.solver_type),
    args.func,
    args.seed,
    args
)

agent = MCTS(
    lb = f.lb,              # the lower bound of each problem dimensions
    ub = f.ub,              # the upper bound of each problem dimensions
    dims = f.dims,          # the problem dimensions
    ninits = 20,      # the number of random samples used in initializations 
    func = f,               # function object to be optimized
    Cp = args.Cp,              # Cp for MCTS
    leaf_size = 20, # tree leaf size
    solver_type = args.solver_type,
    turbo_max_evals = 50,
    kernel_type = 'rbf', #SVM configruation
    gamma_type = 'auto'    #SVM configruation
)

agent.search(max_samples = args.max_samples)

print('best f(x):', agent.value_trace[-1][1])
