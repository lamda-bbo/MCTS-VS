import torch
import botorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import argparse
from benchmark import synthetic_function_problem
from uipt_variable_strategy import UiptRandomStrategy, UiptBestKStrategy
from vanilia_bo import generate_initial_data, get_gpr_model, optimize_acqf
from utils import latin_hypercube, from_unit_cube, save_results


def get_active_idx(dims, active_dims):
    idx = np.random.choice(range(dims), active_dims, replace=False)
    idx = np.sort(idx)
    return idx


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=1000, type=int)
parser.add_argument('--init_samples', default=10, type=int)
parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--active_dims', default=6, type=int)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)

func = synthetic_function_problem[args.func]
dims = func.dims
lb = func.lb
ub = func.ub
uipt_solver = UiptBestKStrategy(dims, k=20)

# train_x, train_y = generate_initial_data(func, args.init_samples, lb, ub)
points = latin_hypercube(args.init_samples, dims)
points = from_unit_cube(points, lb, ub)
train_x, train_y = [], []
for i in range(args.init_samples):
    y = func(points[i])
    train_x.append(points[i])
    train_y.append(y)
sample_counter = args.init_samples
best_y  = [(sample_counter, np.max(train_y))]
uipt_solver.init_strategy(train_x, train_y)

while True:
    # train bo on selected axis
    selected_dims = get_active_idx(dims, args.active_dims)
    selected_x = np.vstack(train_x)[:, selected_dims]
    selected_lb = np.array([lb[idx] for idx in selected_dims])
    selected_ub = np.array([ub[idx] for idx in selected_dims])
    np_train_y = np.array(train_y)
    gpr = get_gpr_model()
    gpr.fit(selected_x, np_train_y)
    selected_new_x, _ = optimize_acqf(args.active_dims, gpr, selected_x, np_train_y, args.batch_size, selected_lb, selected_ub)
    
    # use uipt solver to decide other axis
    X_sample, Y_sample = [], []
    for i in range(len(selected_new_x)):
        fixed_variables = {idx: float(v) for idx, v in zip(selected_dims, selected_new_x[i])}
        new_x = uipt_solver.get_full_variable(fixed_variables, lb, ub)
        new_y = func(new_x)
        X_sample.append(new_x)
        Y_sample.append(new_y)
        uipt_solver.update(new_x, new_y)
    
    train_x.extend(X_sample)
    train_y.extend(Y_sample)
    sample_counter += len(X_sample)
    best_y.append( (sample_counter, np.max(train_y)) )
    if sample_counter >= args.max_samples:
        break

print('best f(x):', best_y[-1][1])
df_data = pd.DataFrame(best_y, columns=['x', 'y'])
save_results('logs', 'dropout{}'.format(args.active_dims), args.func, args.seed, df_data)