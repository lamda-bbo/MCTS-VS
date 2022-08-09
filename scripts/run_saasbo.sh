#!/bin/bash

seed_start=2021
seed_end=2025

func_list=(hartmann6_500)
max_samples=200
root_dir=hartmann6_logs
for func in ${func_list[@]}
do
    echo "Function: " $func
    # lvs-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 saasbo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
done