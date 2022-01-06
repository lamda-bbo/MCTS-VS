#!/bin/bash

seed_start=2024
seed_end=2025
# func_list=(Hopper Walker2d)
func_list=(Hopper )
max_samples=2000
Cp=50
root_dir=rl_logs

for func in ${func_list[@]}
do
    echo "Function: " $func
    # lvs-bo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 lamcts_vs.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --Cp=$Cp \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
    
    # lvs-turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 lamcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=$Cp \
            --ipt_solver=turbo \
            --feature_batch_size=2 \
            --sample_batch_size=3 \
            --min_num_variables=3 \
            --turbo_max_evals=50 \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
    
    # turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 turbo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
    
    # lamcts
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 lamcts.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --Cp=$Cp \
#             --solver_type=turbo \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
done
