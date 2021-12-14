import numpy as np
import matplotlib.pyplot as plt
import argparse
from benchmark import synthetic_function_problem
from uipt_variable_strategy import UiptRandomStrategy, UiptBestKStrategy
from vanilia_bo import generate_initial_data, get_gpr_model, optimize_acqf


def get_active_idx(dims, active_dims):
    idx = np.random.choice(range(dims), active_dims, replace=False)
    idx = np.sort(idx)
    return idx


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str, choices=('hartmann6', 'hartmann6_50', 'hartmann6_100', 'levy10', 'levy10_50', 'levy20', 'levy20_50'))
parser.add_argument('--iterations', default=20, type=int)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

func = synthetic_function_problem['levy10_50']
dims = func.dims
active_dims = 10
lb = func.lb
ub = func.ub

uipt_solver = UiptBestKStrategy(dims, k=20)

train_x, train_y = generate_initial_data(func, dims, lb, ub)
best_y  = [np.max(train_y)]
uipt_solver.init_strategy(train_x, train_y)

for _ in range(70):
    selected_dims = get_active_idx(dims, active_dims)
    selected_x = np.vstack(train_x)[:, selected_dims]
    selected_lb = np.array([lb[idx] for idx in selected_dims])
    selected_ub = np.array([ub[idx] for idx in selected_dims])
    np_train_y = np.array(train_y)
    gpr = get_gpr_model()
    gpr.fit(selected_x, np_train_y)
    selected_new_x, _ = optimize_acqf(active_dims, gpr, selected_x, np_train_y, 3, selected_lb, selected_ub)
    
    X_sample = []
    Y_sample = []
    for i in range(len(selected_new_x)):
        fixed_variables = {idx: float(v) for idx, v in zip(selected_dims, selected_new_x[i])}
        new_x = uipt_solver.get_full_variable(fixed_variables, lb, ub)
        new_y = func(new_x)
        X_sample.append(new_x)
        Y_sample.append(new_y)
        uipt_solver.update(new_x, new_y)
    
    train_x.extend(X_sample)
    train_y.extend(Y_sample)
    best_y.append(np.max(train_y))

print('best func value:', best_y[-1])
plt.plot(best_y)
plt.show()