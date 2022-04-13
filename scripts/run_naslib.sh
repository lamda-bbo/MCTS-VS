#!/bin/bash

seed_start=2021
seed_end=2025
func_list=(nasbenchtrans nasbench201)
# func_list=(nasbench201)
max_samples=200
Cp=0.1
root_dir=naslib_logs

for func in ${func_list[@]}
do
    echo "Function: " $func
    # lvs-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 mcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=$Cp \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # lvs-turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 mcts_vs.py \
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
        }
    done
    
    # lvs-rs
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 mcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=$Cp \
            --ipt_solver=rs \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done

    # random search
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 random_search.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 turbo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # lamcts-turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 lamcts.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=$Cp \
            --solver_type=turbo \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done

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
#         python3 cmaes.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --sigma=0.1 \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait

    # VAE-BO
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 vae_bo.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --update_interval=30 \
#             --active_dims=10 \
#             --lr=0.001 \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
done
