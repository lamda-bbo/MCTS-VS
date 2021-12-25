import numpy as np
import random
import botorch
import torch
import argparse
from benchmark.rl_benchmark import RLEnv


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--seed', default=2021, type=int)
args = parser.parse_args()

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)

f = RLEnv()
print(f(np.random.uniform(-1, 1, 102)))