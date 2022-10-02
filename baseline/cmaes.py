import torch
import botorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
import cma
from benchmark import get_problem
from utils import save_args


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=600, type=int)
parser.add_argument('--pop_size', default=20, type=int)
parser.add_argument('--sigma', default=0.01, type=float)
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
    'algo': 'cmaes',
    'func': args.func,
    'seed': args.seed
}
f = get_problem(args.func, save_config, args.seed)

save_args(
    'config/' + args.root_dir,
    'cmaes',
    args.func,
    args.seed,
    args
)

# ============== cmaes ====================

total_sample = 0
x = np.random.uniform(f.lb, f.ub, f.dims)
cmaes = cma.CMAEvolutionStrategy(x, args.sigma, {'popsize': args.pop_size, 'bounds': [f.lb[0], f.ub[0]], 'seed': args.seed})
# print(cmaes.opts['bounds'])

while total_sample < args.max_samples:
    population = cmaes.ask()
    fitness = []
    for k in range(args.pop_size):
        x = np.array(population[k]).reshape(len(x))
        fitness.append(f(x))
    cmaes.tell(population, np.array(fitness))
    total_sample += args.pop_size
    
print('best f(x):', f.tracker.best_value_trace[-1])