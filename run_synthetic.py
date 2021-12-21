import os
import re
from multiprocessing import Pool

smoke_test = False

if smoke_test:
    func_list = [
        'hartmann6',
        'levy10',
    ]
    max_samples = 100
    seeds = [2022, ]
else:
    func_list = [
        'ackley20_100',
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
    seeds = [2021, 2022, 2023]
    # seeds = [2021, 2022, 2023, 2024, 2025]
    # seeds = [2021, ]

n_processes = 16
root_dir = 'synthetic_logs'
cmds = []
for func in func_list:
    print('test function: {}'.format(func))
    
    # hypterparameters for mcts and dropout
    if func.startswith('ackley20'):
        Cp = 2
        active_dims = 20
    elif func.startswith('branin2'):
        Cp = 1
        active_dims = 2
    elif func.startswith('hartmann6'):
        Cp = 0.1
        active_dims = 6
    elif func.startswith('levy10'):
        Cp = 5
        active_dims = 10
    elif func.startswith('levy20'):
        Cp = 10
        active_dims = 20
    elif func.startswith('rosenbrock20'):
        Cp = 10
        active_dims = 20
    else:
        assert 0, 'Illegal function name'
        
    # vanilia bo
    for seed in seeds:
        cmds.append(
            f'python3 vanilia_bo.py \
                --func={func} \
                --max_samples={max_samples} \
                --root_dir={root_dir} \
                --seed={seed}'
        )
    
    # lamcts variable selection BO
    for seed in seeds:
        cmds.append(
            f'python3 lamcts_vs.py \
                --func={func} \
                --max_samples={max_samples} \
                --Cp={Cp} \
                --ipt_solver=bo \
                --uipt_solver=bestk \
                --root_dir={root_dir} \
                --seed={seed}'
        )
        
    # dropout bo
    for seed in seeds:
        cmds.append(
            f'python3 dropout.py \
                --func={func} \
                --max_samples={max_samples} \
                --active_dims={active_dims} \
                --root_dir={root_dir} \
                --seed={seed}'
        )
        
    # rembo
    for seed in seeds:
        cmds.append(
            f'python3 ax_embedding_bo.py \
                --func={func} \
                --max_samples={max_samples} \
                --active_dims={active_dims} \
                --strategy=rembo \
                --root_dir={root_dir} \
                --seed={seed}'
        )
    
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