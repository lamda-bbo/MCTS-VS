import torch
import botorch
import numpy as np
import pandas as pd
import random
import argparse

from benchmark import synthetic_function_problem
from baseline import Turbo1
from utils import save_results


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=1000, type=int)
parser.add_argument('--root_dir', default='sota_logs', type=str)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)
f = synthetic_function_problem[args.func]


turbo1 = Turbo1(
    f=lambda x: -f(x),  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=20,  # Number of initial bounds from an Latin hypercube design
    max_evals = args.max_samples,  # Maximum number of evaluations
    batch_size=10,  # How large batch size TuRBO uses
    verbose=False,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)

turbo1.optimize()

value = np.minimum.accumulate(turbo1.fX)
value_trace = [(idx+1, -y) for idx, y in enumerate(value.reshape(-1))]

print('best f(x):', value_trace[-1][1])
df_data = pd.DataFrame(value_trace, columns=['x', 'y'])
save_results(args.root_dir, 'turbo1', args.func, args.seed, df_data)