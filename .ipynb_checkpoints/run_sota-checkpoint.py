import os
from multiprocessing import Pool

smoke_test = False

if smoke_test:
    func_list = [
        'hartmann6_50',
        # 'levy10',
    ]
    max_samples = 50
    seeds = [2022, ]
else:
    func_list = [
        # 'ackley20_50',
        # 'branin2_100',
        'hartmann6_100',
        'hartmann6_300',
        'hartmann6_500',
        'levy10_50',
        'levy20_50',
        'rosenbrock20_50'
    ]
    max_samples = 600
    seeds = [2021, 2022, 2023, 2024, 2025]
    
root_dir = 'sota_logs'
cmds = []
for func in func_list:
    print('test function: {}'.format(func))
    
    # hypterparameters for mcts
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
        
    # alebo
    for seed in seeds:
        cmds.append(
            f'python3 ax_embedding_bo.py \
                --func={func} \
                --max_samples={max_samples} \
                --active_dims={active_dims} \
                --strategy=alebo \
                --root_dir={root_dir} \
                --seed={seed}'
        )
        
    # hesbo
    for seed in seeds:
        cmds.append(
            f'python3 ax_embedding_bo.py \
                --func={func} \
                --max_samples={max_samples} \
                --active_dims={active_dims} \
                --strategy=hesbo \
                --root_dir={root_dir} \
                --seed={seed}'
        )
        
    # dropout turbo
    for seed in seeds:
        cmds.append(
            f'python3 dropout.py \
                --func={func} \
                --max_samples={max_samples} \
                --active_dims={active_dims} \
                --ipt_solver=turbo \
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
with Pool() as p:
    p.map(os.system, cmds)
