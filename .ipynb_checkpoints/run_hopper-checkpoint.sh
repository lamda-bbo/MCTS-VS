#!/bin/bash

seed_start=2021
seed_end=2023
# func_list=(HalfCheetah Walker2d)
func_list=(Hopper )
max_samples=10000
root_dir=real_logs

for func in ${func_list[@]}
do
    echo "Function: " $func
    # lvs-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 lamcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=50 \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
    
    # lvs-turbo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 lamcts_vs.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --Cp=50 \
#             --ipt_solver=turbo \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
    
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
#             --Cp=50 \
#             --solver_type=turbo \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
done
