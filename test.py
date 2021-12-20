import numpy as np
import random
import torch
from ax.utils.measurement.synthetic_functions import branin
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.service.managed_loop import optimize
from matplotlib import pyplot as plt


def branin_evaluation_function(parameterization):
    # Evaluates Branin on the first two parameters of the parameterization.
    # Other parameters are unused.
    x = np.array([parameterization["x0"], parameterization["x1"]])
    return {"objective": (branin(x), 0.0)}


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

parameters = [
    {"name": "x0", "type": "range", "bounds": [-5.0, 10.0], "value_type": "float"},
    {"name": "x1", "type": "range", "bounds": [0.0, 15.0], "value_type": "float"},
]
parameters.extend([
    {"name": f"x{i}", "type": "range", "bounds": [-5.0, 10.0], "value_type": "float"}
    for i in range(2, 100)
])
alebo_strategy = ALEBOStrategy(D=100, d=4, init_size=5)
best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    experiment_name="branin_100",
    objective_name="objective",
    evaluation_function=branin_evaluation_function,
    minimize=True,
    total_trials=10,
    generation_strategy=alebo_strategy,
    random_seed=42,
)

objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])

print(objectives[-1])