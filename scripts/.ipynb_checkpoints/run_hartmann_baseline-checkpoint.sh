#!/bin/bash

seed_start=2021
seed_end=2025

func_list=(hartmann6_50 hartmann6_100 hartmann6_300 hartmann6_500)
max_samples=600
Cp=0.1
root_dir=hartmann6_logs

for func in ${func_list[@]}
do
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
        } &
    done
    wait
    
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
        } &
    done
    wait
    
    # cma-es
done
