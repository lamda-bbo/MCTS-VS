import os
import re
from multiprocessing import Pool

func_list = [
    'ackley20_300',
    # 'branin2_100',
    # 'hartmann6_100',
    # 'hartmann6_300',
    # 'hartmann6_500',
    # 'levy10_50',
    # 'levy20_50',
    # 'rosenbrock20_50'
]
max_samples = 600
seeds = list(range(2025, 2025+20))

n_processes = 1
root_dir = 'synthetic_logs'
cmds = []
for func in func_list:
    print('test function: {}'.format(func))
    
    # hypterparameters for mcts and dropout
    Cp = 2
    
    # lamcts
    for seed in seeds:
        cmds.append(
            f'python3 lamcts.py \
                --func={func} \
                --max_samples={max_samples} \
                --Cp={Cp} \
                --solver_type=bo \
                --root_dir={root_dir} \
                --seed={seed}'
        )
    
# run all 
with Pool(processes=n_processes) as p:
    p.map(os.system, cmds)