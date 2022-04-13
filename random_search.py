import torch
import botorch
import numpy as np
import pandas as pd
import argparse
import random
from benchmark import get_problem
from utils import save_args


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=600, type=int)
parser.add_argument('--root_dir', default='synthetic_logs', type=str)
parser.add_argument('--seed', default=2021, type=int)
args = parser.parse_args()

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)

algo_name = 'random_search'
save_config = {
    'save_interval': 50,
    'root_dir': 'logs/' + args.root_dir,
    'algo': algo_name,
    'func': args.func,
    'seed': args.seed
}

save_args(
    'config/' + args.root_dir,
    algo_name,
    args.func,
    args.seed,
    args
)

f = get_problem(args.func, save_config, args.seed)

train_x, train_y, best_y = [], [], []
sample_counter = 0
while True:
    x = np.random.uniform(f.lb, f.ub)
    y = f(x)
    sample_counter += 1
    train_x.append(x)
    train_y.append(y)
    best_y.append( (sample_counter, np.max(train_y)) )
    if sample_counter >= args.max_samples:
        break

print('best f(x):', best_y[-1][1])
