import torch
import botorch
import numpy as np
import pandas as pd
import argparse
import random
from baseline.saasbo import saasbo
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

save_config = {
    'save_interval': 50,
    'root_dir': 'logs/' + args.root_dir,
    'algo': 'saasbo',
    'func': args.func,
    'seed': args.seed
}
f = get_problem(args.func, save_config, args.seed)

save_args(
    'config/' + args.root_dir,
    'saasbo',
    args.func,
    args.seed,
    args
)

X, Y = saasbo.run_saasbo(
    f=lambda x: -f(x),        # function to be minimized
    lb=f.lb,              # lower bounds
    ub=f.ub,              # upper bounds
    max_evals=args.max_samples,       # total evaluation budget
    num_init_evals=10,  # number of initial quasi-random Sobol points
    seed=args.seed,             # controls the seed for the num_init_evals random points
    alpha=0.1,          # controls sparsity in the SAAS prior
    num_warmup=512,     # number of warmup samples used in HMC
    num_samples=256,    # number of post-warmup samples used in HMC
    thinning=16,        # whether to thin the post-warmup samples by some factor
    num_restarts_ei=3,  # number of restarts for EI maximization
    kernel="rbf",       # "rbf" or "matern"
    device="cpu",       # "cpu" or "gpu"
)