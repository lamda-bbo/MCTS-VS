import torch
import botorch
import numpy as np
import pandas as pd
import argparse
import random
from benchmark import get_problem
from utils import latin_hypercube, from_unit_cube, save_results, save_args
from inner_optimizer import run_saasbo_one_epoch

    
parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=600, type=int)
parser.add_argument('--init_samples', default=10, type=int)
parser.add_argument('--batch_size', default=3, type=int)
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
    'algo': 'saasbo',
    'func': args.func,
    'seed': args.seed
}
func = get_problem(args.func, save_config)

save_args(
    'config/' + args.root_dir,
    'saasbo',
    args.func,
    args.seed,
    args
)

dims, lb, ub = func.dims, func.lb, func.ub
neg_func = lambda x: - func(x)

points = latin_hypercube(args.init_samples, dims)
points = from_unit_cube(points, lb, ub)
train_x, train_y = [], []
for i in range(args.init_samples):
    y = neg_func(points[i])
    train_x.append(points[i])
    train_y.append(y)

sample_counter = args.init_samples
best_y  = [(sample_counter, np.max(train_y))]

while True:
    train_x_np = np.array(train_x)
    train_y_np = np.array(train_y)
    proposed_X = run_saasbo_one_epoch(
        train_x_np, 
        train_y_np, 
        dims, 
        args.batch_size,
        neg_func, 
        lb, 
        ub
    )
    proposed_Y = [neg_func(X) for X in proposed_X]
    train_x.extend(proposed_X)
    train_y.extend(proposed_Y)
    sample_counter += len(proposed_X)
    best_y.append( (sample_counter, np.min(train_y)) )
    print('best_y:', best_y[-1])
    if sample_counter >= args.max_samples:
        break

print('best f(x):', best_y[-1][1])
