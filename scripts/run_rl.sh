#!/bin/bash

seed_start=2021
seed_end=2025
func_list=(Hopper Walker2d)
# func_list=(Walker2d)
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
#         python3 mcts_vs.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --Cp=$Cp \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
    
#     # lvs-turbo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 mcts_vs.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --Cp=$Cp \
#             --ipt_solver=turbo \
#             --feature_batch_size=2 \
#             --sample_batch_size=3 \
#             --min_num_variables=3 \
#             --turbo_max_evals=50 \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
    
#     # turbo
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
    
#     # lamcts
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

#     if [ "$func" = "Hopper" ]
#     then
#         active_dims=10
#     elif [ "$func" = "Walker2d" ]
#     then
#         active_dims=20
#     else
#         echo "333"
#     fi

#     # hesbo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 ax_embedding_bo.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --active_dims=$active_dims \
#             --strategy=hesbo \
#             --root_dir=$root_dir \
#             --seed=$seed
#         }
#     done
    
#     # alebo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 ax_embedding_bo.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --active_dims=$active_dims \
#             --strategy=alebo \
#             --root_dir=$root_dir \
#             --seed=$seed
#         }
#     done
    
    # cmaes
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 cmaes.py \
            --func=$func \
            --max_samples=$max_samples \
            --pop_size=50 \
            --sigma=0.01 \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
done
