#!/bin/bash

seed_start=2021
seed_end=2023

# func_list=(levy10_50 levy10_100 levy10_300)
func_list=(levy10_300)
# levt10_100 cp=10 min=5
max_samples=600
Cp=10
root_dir=levy10_logs

for func in ${func_list[@]}
do
    # lvs-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 lamcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --feature_batch_size=2 \
            --sample_batch_size=5 \
            --min_num_variables=3 \
            --Cp=$Cp \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
    
#     # lvs-turbo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 lamcts_vs.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --turbo_max_evals=50 \
#             --min_num_variables=5 \
#             --Cp=$Cp \
#             --ipt_solver=turbo \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
    
#     # vanilla bo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 vanilia_bo.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
    
    # dropout-bo
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
    
    # turbo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 turbo.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
    
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 ax_embedding_bo.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --active_dims=10 \
#             --strategy=hesbo \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
done
