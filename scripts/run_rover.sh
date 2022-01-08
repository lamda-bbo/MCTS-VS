#!/bin/bash

seed_start=2021
seed_end=2023
func_list=(rover)
max_samples=600
Cp=1
root_dir=rover_logs

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
            --Cp=$Cp \
            --min_num_variables=10 \
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
#             --Cp=$Cp \
#             --ipt_solver=turbo \
#             --feature_batch_size=2 \
#             --sample_batch_size=2 \
#             --min_num_variables=10 \
#             --turbo_max_evals=10 \
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
    
    # lamcts
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 lamcts.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --Cp=$Cp \
#             --solver_type=bo \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait

    # hesbo
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
    
    # alebo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 ax_embedding_bo.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --active_dims=10 \
#             --strategy=alebo \
#             --root_dir=$root_dir \
#             --seed=$seed
#         }
#     done
    
    # cma-es
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 ax_embedding_bo.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --active_dims=10 \
#             --strategy=alebo \
#             --root_dir=$root_dir \
#             --seed=$seed
#         }
#     done
done
