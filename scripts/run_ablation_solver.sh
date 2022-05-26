#!/bin/bash

seed_start=2021
seed_end=2025

func_list=(hartmann30_500 hartmann60_500 hartmann90_500 hartmann120_500)
max_samples=600
root_dir=ablation_logs

solver_list=(rs bo turbo)
for func in ${func_list[@]}
do
    for solver in ${solver_list[@]}
    do
        for ((seed=$seed_start; seed<=$seed_end; seed++))
        do
            {
            python3 mcts_vs.py \
                --func=$func \
                --max_samples=$max_samples \
                --turbo_max_evals=50 \
                --Cp=0.1 \
                --ipt_solver=$solver \
                --dir_name=${func}_solver \
                --root_dir=$root_dir \
                --seed=$seed
            } &
        done
        wait
    done
done