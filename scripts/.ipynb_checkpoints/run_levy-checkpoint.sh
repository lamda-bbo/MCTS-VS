#!/bin/bash

seed_start=2021
seed_end=2023

func_list=(levy10_50 levy20_50)
max_samples=600
Cp=10
root_dir=sota_logs

for func in ${func_list[@]}
do
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 lamcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=$Cp \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
    
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 lamcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --turbo_max_evals=50 \
            --Cp=$Cp \
            --ipt_solver=turbo \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
    
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
    
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 dropout.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --active_dims=10 \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
done
