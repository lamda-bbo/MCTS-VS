import os
from multiprocessing import Pool

smoke_test = False

if smoke_test:
    func_list = [
        'hartmann6',
        'levy10',
    ]
    max_samples = 100
    seeds = [42, 43]
else:
    func_list = [
        # 'hartmann6',
        'hartmann6_50',
        'hartmann6_100',
        'hartmann6_300',
        'hartmann6_500',
        # 'levy10',
        'levy10_50',
        'levy10_100',
        'levy10_300',
        'levy10_500',
        # 'levy20',
        'levy20_50',
        'levy20_100',
        'levy20_300',
        'levy20_500',
    ]
    max_samples = 1000
    seeds = [42, 43, 44]
    
root_dir = 'sota_logs'

cmds = []
for func in func_list:
    print('test function: {}'.format(func))
    
    # hypterparameters for mcts
    if func.startswith('hartmann6'):
        Cp = 0.1
    elif func.startswith('levy10'):
        Cp = 5
    elif func.startwith('levy20'):
        Cp = 10
    
    # lamcts variable selection
    for seed in seeds:
        cmds.append(
            f'python3 lamcts_vs.py \
                --func={func} \
                --max_samples={max_samples} \
                --turbo_max_evals=50 \
                --Cp={Cp} \
                --ipt_solver=turbo \
                --uipt_solver=bestk \
                --root_dir={root_dir} \
                --seed={seed}'
        )
    
    # turbo1
    for seed in seeds:
        cmds.append(
            f'python3 turbo.py \
                --func={func} \
                --max_samples={max_samples} \
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
                --solver_type=turbo \
                --root_dir={root_dir} \
                --seed={seed}'
        )
        
# run all 
with Pool(processes=32) as p:
    p.map(os.system, cmds)
