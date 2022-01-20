#!/bin/bash

seed_start=2021
seed_end=2030

func_list=(hartmann6_300 hartmann6_500)
# func_list=(hartmann6_100 hartmann6_1000)
max_samples=100
Cp=0.1
root_dir=time_logs

for func in ${func_list[@]}
do
    # lvs-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 lamcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=$Cp \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
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
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 vanilia_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # dropout-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 dropout.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=6 \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done

    # dropout-turbo
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 dropout.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --active_dims=6 \
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

#     # lamcts-turbo
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
#         }
#     done
    
    # rembo
    # for ((seed=$seed_start; seed<=$seed_end; seed++))
    # do
    #     {
    #     python3 ax_embedding_bo.py \
    #         --func=$func \
    #         --max_samples=$max_samples \
    #         --active_dims=6 \
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
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 ax_embedding_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=6 \
            --strategy=hesbo \
            --root_dir=$root_dir \
            --seed=$seed
        } 
    done
    
    # alebo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 ax_embedding_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=6 \
            --strategy=alebo \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # cmaes
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 cmaes.py \
            --func=$func \
            --max_samples=$max_samples \
            --pop_size=20 \
            --sigma=0.8 \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
done
