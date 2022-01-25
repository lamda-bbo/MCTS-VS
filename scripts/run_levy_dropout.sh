#!/bin/bash

seed_start=2021
seed_end=2025

func_list=(levy10_50 levy10_100 levy10_300)
dim_list=(3 6 10 15 20 30)
max_samples=600
root_dir=dropout_logs

for func in ${func_list[@]}
do
    # dropout-bo
    for active_dims in ${dim_list[@]}
    do
        for ((seed=$seed_start; seed<=$seed_end; seed++))
        do
            {
            python3 dropout.py \
                --func=$func \
                --max_samples=$max_samples \
                --active_dims=$active_dims \
                --root_dir=$root_dir \
                --seed=$seed
            } &
        done
        wait
    done
done