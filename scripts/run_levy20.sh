#!/bin/bash

seed_start=2021
seed_end=2025

func_list=(levy20_50 levy20_100 levy20_300)
max_samples=600
Cp=10
root_dir=levy20_logs

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
done
