# Monte Carlo Tree Search based Variable Selection for High-Dimensional Bayesian Optimization

This package includes the Python code of the MCTS-VS algorithm. MCTS-VS is a new high-dimensional Bayesian optimization algorithm. MCTS-VS employed MCTS to partition the variables into important and unimportant ones, and only those selected important variables are optimized via any black-box optimization algorithm, e.g., vanilla BO or TuRBO.

## Requirements

## Usage

Run the scripts in the ```./scripts/``` directory. For example, run ```bash scripts/run_hartmann.sh``` to evaluate MCTS-VS and other baselines on Hartmann.



For Simplicity, all hyper-parameter settings are included in the run_experiments.sh file. To evaluate
SGES on the four RL tasks in the paper, you only need run
