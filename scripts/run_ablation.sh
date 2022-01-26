#!/bin/bash

seed_start=2021
seed_end=2025

func=hartmann6_300
max_samples=600
root_dir=ablation_logs

# fill-in strategy
# strategy_list=(bestk random copy mix)
# for strategy in ${strategy_list[@]}
# do
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 lamcts_vs.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --Cp=0.1 \
#             --uipt_solver=$strategy \
#             --postfix=$strategy \
#             --dir_name=strategy \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
# done

# # Cp
root_dir=ablation_logs
Cp_list=(0.01 0.1 1 10 100)
func_list=(hartmann6_300 hartmann6_500 levy10_100 levy10_300)
for func in ${func_list[@]}
do
    for Cp in ${Cp_list[@]}
    do
        for ((seed=$seed_start; seed<=$seed_end; seed++))
        do
            {
            python3 lamcts_vs.py \
                --func=$func \
                --max_samples=$max_samples \
                --Cp=$Cp \
                --uipt_solver=bestk \
                --postfix=$Cp \
                --dir_name=${func}_Cp \
                --root_dir=$root_dir \
                --seed=$seed
            } &
        done
        wait
    done
done

# min_num_variable
# min_num_variables_list=(3 6 10 20 50)
# for min_num_variables in ${min_num_variables_list[@]}
# do
#     for ((seed=$seed_start; seed<=$seed_end; seed++))
#     do
#         {
#         python3 lamcts_vs.py \
#             --func=$func \
#             --max_samples=$max_samples \
#             --min_num_variables=$min_num_variables \
#             --Cp=0.1 \
#             --uipt_solver=bestk \
#             --postfix=$min_num_variables \
#             --dir_name=min_num_variables \
#             --root_dir=$root_dir \
#             --seed=$seed
#         } &
#     done
#     wait
# done

# number of samples
# feature_bs_list=(2 5)
# sample_bs_list=(3 5 10)
# for f_bs in ${feature_bs_list[@]}
# do
#     for s_bs in ${sample_bs_list[@]}
#     do
#         for ((seed=$seed_start; seed<=$seed_end; seed++))
#         do
#             {
#             python3 lamcts_vs.py \
#                 --func=$func \
#                 --max_samples=$max_samples \
#                 --feature_batch_size=$f_bs \
#                 --sample_batch_size=$s_bs \
#                 --Cp=0.1 \
#                 --uipt_solver=bestk \
#                 --postfix=${f_bs}_${s_bs} \
#                 --dir_name=num_samples \
#                 --root_dir=$root_dir \
#                 --seed=$seed
#             } &
#         done
#         wait
#     done
# done

# k of best-k
k_list=(1 5 10 15 20)
for k in ${k_list[@]}
do
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 lamcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=0.1 \
            --k=$k \
            --uipt_solver=bestk \
            --postfix=$k \
            --dir_name=param_k \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
done