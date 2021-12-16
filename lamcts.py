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
from baseline import MCTS
from benchmark import synthetic_function_problem
from utils import save_results


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=1000, type=int)
parser.add_argument('--Cp', default=0.1, type=float)
parser.add_argument('--solver_type', default='bo', type=str)
parser.add_argument('--root_dir', default='logs', type=str)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)
f = synthetic_function_problem[args.func]

args = parser.parse_args()

agent = MCTS(
    lb = f.lb,              # the lower bound of each problem dimensions
    ub = f.ub,              # the upper bound of each problem dimensions
    dims = f.dims,          # the problem dimensions
    ninits = 20,      # the number of random samples used in initializations 
    func = f,               # function object to be optimized
    Cp = args.Cp,              # Cp for MCTS
    leaf_size = 20, # tree leaf size
    solver_type=args.solver_type,
    kernel_type = 'rbf', #SVM configruation
    gamma_type = 'auto'    #SVM configruation
)

agent.search(max_samples = args.max_samples)

print('best f(x):', agent.value_trace[-1][1])
df_data = pd.DataFrame(agent.value_trace, columns=['x', 'y'])
save_results(args.root_dir, 'lamcts_{}'.format(args.solver_type), args.func, args.seed, df_data)