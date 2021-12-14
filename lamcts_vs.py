import torch
import botorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
from benchmark import synthetic_function_problem
from MCTS import MCTS
from utils import save_results


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=1000, type=int)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)
f = synthetic_function_problem[args.func]

agent = MCTS(
    func=f,
    dims=f.dims,
    lb=f.lb,
    ub=f.ub,
)

agent.search(max_samples=args.max_samples, verbose=False)

print('best f(x):', agent.value_trace[-1][1])
df_data = pd.DataFrame(agent.value_trace, columns=['x', 'y'])
save_results('logs', 'lamcts_vs', args.func, args.seed, df_data)