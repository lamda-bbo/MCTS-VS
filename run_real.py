import os
from multiprocessing import Pool

smoke_test = False

if smoke_test:
    func_list = [
        'nasbench',
        'rover'
    ]
    max_samples = 50
    seeds = [2022, ]
else:
    func_list = [
        'nasbench',
        # 'rover'
    ]
    max_samples = 2000
    seeds = [2021, 2022, 2023]
    # seeds = [2021, 2022, 2023, 2024, 2025]
    
n_processes = 1
root_dir = 'real_logs'
cmds = []
for func in func_list:
    print('test function: {}'.format(func))
    
    if func == 'nasbench':
        Cp = 0.1
    elif func == 'rover':
        Cp = 1
    else:
        assert 0, 'Illegal function name'
        
#     # lamcts variable selection BO
#     for seed in seeds:
#         cmds.append(
#             f'python3 lamcts_vs.py \
#                 --func={func} \
#                 --max_samples={max_samples} \
#                 --Cp={Cp} \
#                 --ipt_solver=bo \
#                 --uipt_solver=bestk \
#                 --root_dir={root_dir} \
#                 --seed={seed}'
#         )
        
#     # vanilia bo
#     for seed in seeds:
#         cmds.append(
#             f'python3 vanilia_bo.py \
#                 --func={func} \
#                 --max_samples={max_samples} \
#                 --root_dir={root_dir} \
#                 --seed={seed}'
#         )
        
#     # dropout bo
#     # for active_dims in [9, 12, 15]:
#     for active_dims in [15, ]:
#         for seed in seeds:
#             cmds.append(
#                 f'python3 dropout.py \
#                     --func={func} \
#                     --max_samples={max_samples} \
#                     --active_dims={active_dims} \
#                     --root_dir={root_dir} \
#                     --seed={seed}'
#             )

    # lamcts vs turbo
    # rover hyperparameter setting
#     for seed in seeds:
#         cmds.append(
#             f'python3 lamcts_vs.py \
#                 --func={func} \
#                 --max_samples={max_samples} \
#                 --feature_batch_size=1 \
#                 --sample_batch_size=3 \
#                 --min_num_variables=20 \
#                 --select_right_threshold=10 \
#                 --turbo_max_evals=100 \
#                 --Cp={Cp} \
#                 --ipt_solver=turbo \
#                 --uipt_solver=bestk \
#                 --root_dir={root_dir} \
#                 --seed={seed}'
#         )
    
    # nasbench
    for seed in seeds:
        cmds.append(
            f'python3 lamcts_vs.py \
                --func={func} \
                --max_samples={max_samples} \
                --feature_batch_size=1 \
                --sample_batch_size=3 \
                --min_num_variables=18 \
                --select_right_threshold=10 \
                --turbo_max_evals=100 \
                --Cp={Cp} \
                --ipt_solver=turbo \
                --uipt_solver=bestk \
                --root_dir={root_dir} \
                --seed={seed}'
        )
            
#     # turbo
#     for seed in seeds:
#         cmds.append(
#             f'python3 turbo.py \
#                 --func={func} \
#                 --max_samples={max_samples} \
#                 --root_dir={root_dir} \
#                 --seed={seed}'
#         )
        
# run all 
# with Pool(processes=n_processes) as p:
#     p.map(os.system, cmds)
for cmd in cmds:
    os.system(cmd)
    