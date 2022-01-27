#!/bin/bash

seed_start=2021
seed_end=2050

# func_list=(levy10_50 levy10_100)
func_list=(levy10_100 levy10_300)
max_samples=100
Cp=10
root_dir=time_logs

for func in ${func_list[@]}
do
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
#         }
#     done
    
    # lvs-turbo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 lamcts_vs.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --turbo_max_evals=50 \
#             --Cp=$Cp \
#             --ipt_solver=turbo \
#             --root_dir=$root_dir \
#             --seed=$seed
#         }
#     done
    
    # vanilla bo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 vanilia_bo.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --root_dir=$root_dir \
#             --seed=$seed
#         }
#     done
    
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
#         }
#     done

    # dropout-turbo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 dropout.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --active_dims=10 \
#             --ipt_solver=turbo \
#             --root_dir=$root_dir \
#             --seed=$seed
#         }
#     done
    
#     # lamcts-bo
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
#         }
#     done

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
    
    # rembo
    # for ((seed=$seed_start; seed<=$seed_end; seed++))
    # do
    #     {
    #     python3 ax_embedding_bo.py \
    #         --func=$func \
    #         --max_samples=$max_samples \
    #         --active_dims=10 \
    #         --strategy=rembo \
    #         --root_dir=$root_dir \
    #         --seed=$seed
    #     }
    # done
    
    # turbo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 turbo.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --root_dir=$root_dir \
#             --seed=$seed
#         }
#     done
    
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
#         } 
#     done
    
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
    
    # cmaes
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 cmaes.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --pop_size=20 \
#             --sigma=10.0 \
#             --root_dir=$root_dir \
#             --seed=$seed
#         }
#     done
done
