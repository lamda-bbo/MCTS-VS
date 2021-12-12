import torch
import botorch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from benchmark_problem import hartmann6, hartmann6_50, hartmann6_100, levy10, levy10_50, levy20, levy20_50
from MCTS import MCTS


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6', type=str, choices=('hartmann6', 'hartmann6_50', 'hartmann6_100', 'levy10', 'levy10_50', 'levy20', 'levy20_50'))
parser.add_argument('--iterations', default=100, type=int)
args = parser.parse_args()

np.random.seed(42)
botorch.manual_seed(42)
torch.manual_seed(42)
f = eval(args.func)

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
    
num_sample, best_value = zip(*agent.best_value_trace)
plt.plot(num_sample, best_value)
plt.savefig('f.png')
plt.show()

print(feature_cnt)

print(agent.ROOT.axis_score)
print(np.argsort(agent.ROOT.axis_score))
plt.bar(list(range(len(agent.ROOT.axis_score))), agent.ROOT.axis_score)
plt.savefig('score.png')
plt.show()

# feature_select_freq = np.zeros(f.dims)
# score_sum = np.zeros(f.dims)
# for feature, score in agent.features:
#     feature_select_freq += feature
#     score_sum += feature * score
# print(feature_select_freq)
# print(score_sum)
# print(score_sum / feature_select_freq)