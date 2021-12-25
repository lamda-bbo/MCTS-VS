#!/bin/bash

seed_start=2021
seed_end=2023

func_list=(rosenbrock10_100 rosenbrock10_300)
max_samples=600
root_dir=sota_logs

for func in ${func_list[@]}
do
    Cp=10
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 lamcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=$Cp \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
    
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
    
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 dropout.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=10 \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
done
