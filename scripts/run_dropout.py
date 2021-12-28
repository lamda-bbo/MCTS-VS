import os
import re
from multiprocessing import Pool

smoke_test = True

if smoke_test:
    func_list = [
        'hartmann6',
        'levy10',
    ]
    max_samples = 100
    seeds = [2021, 2022]
else:
    func_list = [
        'hartmann6_100',
        'hartmann6_300',
        'hartmann6_500',
        'levy10_50',
        'levy20_50',
    ]
    max_samples = 600
    seeds = [2021, 2022, 2023, 2024, 2025]

n_processes = 16
root_dir = 'dropout_logs'
cmds = []
for func in func_list:
    print('test function: {}'.format(func))
        
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
        for seed in seeds:
            cmds.append(
                f'python3 dropout.py \
                    --func={func} \
                    --max_samples={max_samples} \
                    --active_dims={active_dims} \
                    --root_dir={root_dir} \
                    --seed={seed}'
            )
            
# run all 
with Pool(processes=n_processes) as p:
    p.map(os.system, cmds)