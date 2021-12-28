#!/bin/bash

seed_start=2021
seed_end=2026

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
        python3 lamcts_vs.py \
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
        python3 lamcts_vs.py \
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
    
    # vanilla bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 vanilia_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
    
    # dropout-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 dropout.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=20 \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
    
    # lamcts-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 lamcts.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=$Cp \
            --solver_type=bo \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
    
    # rembo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 ax_embedding_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=20 \
            --strategy=rembo \
            --root_dir=$root_dir \
            --seed=$seed
        } &
    done
    wait
done
