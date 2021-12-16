import os
import re
from multiprocessing import Pool

smoke_test = False

if smoke_test:
    func_list = [
        'hartmann6',
        # 'hartmann6_50',
        'levy10',
    ]
    max_samples = 100
    seeds = [42, 43]
else:
    func_list = [
        'hartmann6',
        'hartmann6_50',
        'hartmann6_100',
        'hartmann6_300',
        'hartmann6_500',
        'levy10',
        'levy10_50',
        'levy10_100',
        'levy10_300',
        'levy10_500',
        'levy20',
        'levy20_50',
        'levy20_100',
        'levy20_300',
        'levy20_500',
    ]
    max_samples = 1000
    seeds = [42, 43, 44]
    
root_dir = 'simple_logs'
cmds = []

for func in func_list:
    print('test function: {}'.format(func))
    
    # hypterparameters for mcts
    if func.startswith('hartmann6'):
        Cp = 0.1
    elif func.startswith('levy10'):
        Cp = 5
    elif func.startswith('levy20'):
        Cp = 10
    
    # lamcts variable selection 
    
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
        
with Pool() as p:
    p.map(os.system, cmds)
    