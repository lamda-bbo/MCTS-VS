import torch
import botorch
import numpy as np
import pandas as pd
import random
import argparse
from benchmark import get_problem
from uipt_variable_strategy import UiptRandomStrategy, UiptBestKStrategy
from baseline.vanilia_bo import generate_initial_data, get_gpr_model, optimize_acqf
from utils import latin_hypercube, from_unit_cube, save_results, save_args
from inner_optimizer import Turbo1_VS_Component


def get_active_idx(dims, active_dims):
    idx = np.random.choice(range(dims), active_dims, replace=False)
    idx = np.sort(idx)
    return idx


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=600, type=int)
parser.add_argument('--turbo_max_evals', default=50, type=int)
parser.add_argument('--init_samples', default=10, type=int)
parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--active_dims', default=6, type=int)
parser.add_argument('--ipt_solver', default='bo', type=str)
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
    'algo': 'dropout_{}_{}'.format(args.ipt_solver, args.active_dims),
    'func': args.func,
    'seed': args.seed
}
func = get_problem(args.func, save_config)
dims = func.dims
lb = func.lb
ub = func.ub
uipt_solver = UiptBestKStrategy(dims, k=20)

save_args(
    'config/' + args.root_dir,
    'dropout_{}_{}'.format(args.ipt_solver, args.active_dims),
    args.func,
    args.seed,
    args
)

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
    
    X_sample, Y_sample = [], []
    if args.ipt_solver == 'bo':
        gpr = get_gpr_model()
        gpr.fit(selected_x, np_train_y)
        selected_new_x, _ = optimize_acqf(args.active_dims, gpr, selected_x, np_train_y, args.batch_size, selected_lb, selected_ub)

        # use uipt solver to decide other axis
        for i in range(len(selected_new_x)):
            fixed_variables = {idx: float(v) for idx, v in zip(selected_dims, selected_new_x[i])}
            new_x = uipt_solver.get_full_variable(fixed_variables, lb, ub)
            new_y = func(new_x)
            X_sample.append(new_x)
            Y_sample.append(new_y)
            uipt_solver.update(new_x, new_y)
    elif args.ipt_solver == 'turbo':
        turbo1 = Turbo1_VS_Component(
            f  = lambda x: -func(x),              # Handle to objective function
            lb = selected_lb,           # Numpy array specifying lower bounds
            ub = selected_ub,           # Numpy array specifying upper bounds
            n_init = 1,            # unused parameter
            max_evals  = args.turbo_max_evals, # Maximum number of evaluations
            batch_size = 10,         # How large batch size TuRBO uses
            verbose=False,           # Print information from each batch
            use_ard=True,           # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000, # When we switch from Cholesky to Lanczos
            n_training_steps=50,    # Number of steps of ADAM to learn the hypers
            min_cuda=1024,          #  Run on the CPU for small datasets
            device="cpu",           # "cpu" or "cuda"
            dtype="float32",        # float64 or float32
        )

        Y_init = -np_train_y
        X_sample, Y_sample = turbo1.optimize(selected_x, Y_init, selected_dims, uipt_solver, n=1)
        Y_sample = [-y for y in Y_sample]
    elif args.ipt_solver == 'rs':
        selected_new_x = []
        for _ in range(args.batch_size):
            ipt_x = []
            for i in range(len(selected_dims)):
                ipt_x.append(np.random.uniform(selected_lb[i], selected_ub[i]))
            selected_new_x.append(np.array(ipt_x))
        # use uipt solver to decide other axis
        for i in range(len(selected_new_x)):
            fixed_variables = {idx: float(v) for idx, v in zip(selected_dims, selected_new_x[i])}
            new_x = uipt_solver.get_full_variable(fixed_variables, lb, ub)
            new_y = func(new_x)
            X_sample.append(new_x)
            Y_sample.append(new_y)
            uipt_solver.update(new_x, new_y)
    elif args.ipt_solver == 'smac':
        pass
    else:
        assert 0
    
    train_x.extend(X_sample)
    train_y.extend(Y_sample)
    sample_counter += len(X_sample)
    best_y.append( (sample_counter, np.max(train_y)) )
    if sample_counter >= args.max_samples:
        break

print('best f(x):', best_y[-1][1])
