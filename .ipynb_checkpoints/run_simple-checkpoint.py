import os
import re
from multiprocessing import Pool

smoke_test = True

if smoke_test:
    func_list = [
        'hartmann6',
        'hartmann6_50',
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
    cmds = []
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
    with Pool(processes=len(seeds)) as p:
        p.map(os.system, cmds)
        
    # vanilia bo
    cmds = []
    for seed in seeds:
        cmds.append(
            f'python3 vanilia_bo.py \
                --func={func} \
                --max_samples={max_samples} \
                --root_dir={root_dir} \
                --seed={seed}'
        )
    with Pool(processes=len(seeds)) as p:
        p.map(os.system, cmds)
        
    # dropout bo
    valid_dims = re.findall(r'\d+', func.split('_')[0])
    assert len(valid_dims) == 1, 'Illegal function name'
    valid_dims = int(valid_dims[0])
    if valid_dims == 6:
        active_dims_list = [3, 6, 10]
    elif valid_dims == 10:
        active_dims_list = [6, 10, 15]
    elif valid_dims == 20:
        active_dims_list = [15, 20, 25]
    else:
        assert 0, 'Undefined valid dims'
    
    for active_dims in active_dims_list:
        cmds = []
        for seed in seeds:
            cmds.append(
                f'python3 dropout.py \
                    --func={func} \
                    --max_samples={max_samples} \
                    --active_dims={active_dims} \
                    --root_dir={root_dir} \
                    --seed={seed}'
            )
        with Pool(processes=len(seeds)) as p:
            p.map(os.system, cmds)
    