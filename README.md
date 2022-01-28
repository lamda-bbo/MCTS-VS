# Monte Carlo Tree Search based Variable Selection for High-Dimensional Bayesian Optimization

This package includes the Python code of the MCTS-VS algorithm. MCTS-VS is a new high-dimensional Bayesian optimization algorithm. MCTS-VS employed MCTS to partition the variables into important and unimportant ones, and only those selected important variables are optimized via any black-box optimization algorithm, e.g., vanilla BO or TuRBO.

## Requirements

- Ubuntu 18.04
- Python == 3.8.8
- PyTorch == 1.10.1
- ax == 0.2.2
- BoTorch == 0.5.1
- cma == 3.1.0
- [NAS-Bench-101](https://github.com/google-research/nasbench)

## Usage

Run the scripts in the ```./scripts/``` directory. All hyper-parameters of different algorithms can be found in the scripts. For example, run ```bash scripts/run_hartmann.sh``` to evaluate MCTS-VS and other baselines on Hartmann.

## Source Code

- ```mcts_vs.py``` and ```MCTSVS``` are the main code implement of MCTS-VS algorithm.
- ```uipt_variable_strategy.py``` is the implement of the "fill-in" strategy. 
