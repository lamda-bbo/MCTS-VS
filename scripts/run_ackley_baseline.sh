#!/bin/bash

seed_start=2021
seed_end=2026

func_list=(ackley20_100 ackley20_300)
max_samples=1000
Cp=2
root_dir=ackley20_logs

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
            --active_dims=20 \
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
            --active_dims=20 \
            --strategy=alebo \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
    
    # cma-es
done