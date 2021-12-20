import torch
import botorch
import ax
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import REMBOStrategy, HeSBOStrategy
from ax.service.managed_loop import optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
from benchmark import synthetic_function_problem
from utils import save_results

from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace

from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.benchmark.benchmark import full_benchmark_run, benchmark_replication


class Hartmann6Metric(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return func(x)


def evaluation_function(parameterization):
    x = np.array([parameterization['x'+str(i)] for i in range(dims)])
    return {'objective': (f(x), 0.0)}


hartmann6_100 = BenchmarkProblem(
    name="Hartmann6, D=50",
    optimal_value=-3.32237,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=Hartmann6Metric(
                name="objective",
                param_names=['x' + str(i) for i in range(50)],
                noise_sd=0.0,
            ),
            minimize=True,
        )
    ),
    search_space=SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
            )
            for i in range(50)
        ]
    ),
)




parser = argparse.ArgumentParser()
parser.add_argument('--func', default='hartmann6_50', type=str)
parser.add_argument('--max_samples', default=1000, type=int)
parser.add_argument('--strategy', default='rembo', type=str)
parser.add_argument('--root_dir', default='simple_logs', type=str)
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)

func = synthetic_function_problem[args.func]
dims = func.dims
valid_dims = len(func.valid_idx)

parameters = [
    {'name': f'x{i}', 'type': 'range', 'bounds': [func.lb[i], func.ub[i]], 'value_type': 'float'} for i in range(dims)
]

if args.strategy == 'rembo':
    embedding_strategy = REMBOStrategy(D=dims, d=valid_dims, init_per_proj=2)
elif args.strategy == 'hesbo':
    embedding_strategy = HeSBOStrategy(D=dims, d=valid_dims, init_per_proj=2)
elif args.strategy == 'alebo':
    embedding_strategy = ALEBOStrategy(D=dims, d=valid_dims, init_size=10)
else:
    assert 0, 'Strategy should be rembo, hesbo, alebo'

# TODO: fix the seed

# all_benchmarks = full_benchmark_run(
#     num_replications=1,  # Running them 1 at a time for distributed
#     num_trials=20,
#     batch_size=1,
#     method_groups={'emb': [embedding_strategy, ] },
#     problem_groups={'emb': [hartmann6_100, ]},
#     verbose_logging=True,
# )

result = benchmark_replication(problem=hartmann6_100, method=embedding_strategy, num_trials=20, replication_index=10)

print(result)

objectives = np.array([trial.objective_mean for trial in result.trials.values()])


# objectives = np.array([trial.objective_mean for trial in all_benchmarks['Hartmann6, D=50']['REMBO'][0].trials.values()])

best_value = np.maximum.accumulate(objectives)

print(best_value[-1])

# plt.plot(range(len(best_value)), best_value)
# plt.savefig('./tmp.png')