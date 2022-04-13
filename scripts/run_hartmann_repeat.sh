#!/bin/bash

seed_start=2021
seed_end=2021

func_list=(hartmann6_500)
max_samples=50
Cp=0.1
root_dir=repeat_logs

for func in ${func_list[@]}
do
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
        } &
    done
    wait
    
    # lvs-turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 mcts_vs.py \
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
        } &
    done
    wait
    
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
done
