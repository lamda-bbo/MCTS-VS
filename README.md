# Monte Carlo Tree Search based Variable Selection for High-Dimensional Bayesian Optimization

This repository contains the Python code for MCTS-VS, an algorithm for high-dimensional Bayesian optimization described in [Monte Carlo Tree Search based Variable Selection for High-Dimensional Bayesian Optimization](https://arxiv.org/abs/2210.01628). 

MCTS-VS employed MCTS to partition the variables into important and unimportant ones, and only those selected important variables are optimized via any black-box optimization algorithm, e.g., vanilla BO or TuRBO.

## Poster

![poster](https://user-images.githubusercontent.com/19886779/223073427-6f2a2978-ce55-44b5-9af3-cda381dda582.png)


## Requirements

- Ubuntu == 18.04
- Python == 3.8.8
- PyTorch == 1.10.1
- ax == 0.2.2
- BoTorch == 0.5.1
- cma == 3.1.0
- [NAS-Bench-101](https://github.com/google-research/nasbench)
- NAS-Bench-1Shot1 in [HPO-Bench](https://github.com/automl/HPOBench)
- NAS-Bench-201, TransNAS-Bench-101, NAS-Bench-ASR in [NASLib](https://github.com/automl/NASLib)

## File structure

- ```benchmark```  directory is the implement of the benchmark problems. 
- ```mcts_vs.py``` and ```MCTSVS``` directory are the main code implement of MCTS-VS algorithm.
- ```inner_optimizer``` is the implement of the optimizer used for the selected variables. 
- ```uipt_variable_strategy.py``` is the implement of the "fill-in" strategy. 
- ```baseline``` directory is the implement of all baseline algorithms. 

## Usage

Run ```bash scripts/run_hartmann6.sh``` to evaluate MCTS-VS and other baselines on Hartmann function. 

## Citation

```
@inproceedings{MCTSVS,
    author = {Lei Song, Ke Xue, Xiaobin Huang, Chao Qian},
    title = {{M}onte {C}arlo Tree Search based Variable Selection for High-Dimensional {B}ayesian Optimization},
    booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS'22)},
    year = {2022},
    address={New Orleans, LA}
}
```
