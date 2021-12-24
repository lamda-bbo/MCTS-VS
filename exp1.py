import torch
import botorch
import numpy as np
import pandas as pd
import random
import argparse
from benchmark import get_problem
from vanilia_bo import generate_initial_data, get_gpr_model, optimize_acqf
from utils import latin_hypercube, from_unit_cube, save_results


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_300', type=str)
parser.add_argument('--max_samples', default=600, type=int)
parser.add_argument('--init_samples', default=1, type=int)
parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--root_dir', default='exp1', type=str)
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
    'algo': 'fixed_{}'.format(1),
    'func': args.func,
    'seed': args.seed
}
func = get_problem(args.func, save_config)
dims = func.dims
lb = func.lb
ub = func.ub

points = latin_hypercube(args.init_samples, dims)
points = from_unit_cube(points, lb, ub)
train_x, train_y = [], []
for i in range(args.init_samples):
    y = func(points[i])
    train_x.append(points[i])
    train_y.append(y)
sample_counter = args.init_samples
best_y  = [(sample_counter, np.max(train_y))]

# epoch14: 27: [0, 20, 24, 28, 34, 42, 49, 55, 69, 96, 100, 101, 108, 111, 113, 126, 137, 147, 155, 167, 169, 177, 194, 259, 273, 290, 291]

selected_dims = [0, 20, 24, 28, 34, 42, 49, 55, 69, 96, 100, 101, 108, 111, 113, 126, 137, 147, 155, 167, 169, 177, 194, 259, 273, 290, 291]
unselected_dims = list(set(range(dims)) - set(selected_dims))

unselected_variables = {idx: np.random.uniform(lb[idx], ub[idx]) for idx in unselected_dims}

while True:
    selected_x = np.vstack(train_x)[:, selected_dims]
    selected_lb = np.array([lb[idx] for idx in selected_dims])
    selected_ub = np.array([ub[idx] for idx in selected_dims])
    np_train_y = np.array(train_y)
    X_sample, Y_sample = [], []
    
    gpr = get_gpr_model()
    gpr.fit(selected_x, np_train_y)
    selected_new_x, _ = optimize_acqf(len(selected_dims), gpr, selected_x, np_train_y, args.batch_size, selected_lb, selected_ub)
    
    for i in range(len(selected_new_x)):
        new_x = np.zeros(dims)
        for idx in range(dims):
            selected_variables = {idx: float(v) for idx, v in zip(selected_dims, selected_new_x[i])}
            if idx in selected_dims:
                new_x[idx] = selected_variables[idx]
            else:
                new_x[idx] = unselected_variables[idx]
        new_y = func(new_x)
        X_sample.append(new_x)
        Y_sample.append(new_y)
        
    train_x.extend(X_sample)
    train_y.extend(Y_sample)
    sample_counter += len(X_sample)
    if sample_counter >= args.max_samples:
        break
        
                