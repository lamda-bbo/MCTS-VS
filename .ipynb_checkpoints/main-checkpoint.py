import torch
import botorch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from benchmark import synthetic_function_problem
from MCTS import MCTS


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str, choices=('hartmann6', 'hartmann6_50', 'hartmann6_100', 'levy10', 'levy10_50', 'levy20', 'levy20_50'))
parser.add_argument('--iterations', default=20, type=int)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

print(args)

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

agent.search(iterations=args.iterations)

feature_cnt = np.zeros(f.dims)
for feature in agent.ROOT.features:
    feature_cnt += feature
    
num_sample, best_value = zip(*agent.value_trace)
plt.figure()
plt.plot(num_sample, best_value)
plt.savefig('sample_cnt_f.png')
plt.show()

# print(feature_cnt)

# print(agent.ROOT.axis_score)
# print(np.argsort(agent.ROOT.axis_score))
# plt.figure()
# plt.bar(list(range(len(agent.ROOT.axis_score))), agent.ROOT.axis_score)
# plt.savefig('score.png')
# plt.show()
