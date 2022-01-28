#!/bin/bash

seed_start=2021
seed_end=2025
max_samples=600

func_list=(hartmann6_100 hartmann6_300 hartmann6_500)

for func in ${func_list[@]}
do
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        python3 mcts_vs_theory.py \
            --seed=$seed \
            --func=$func \
            --max_samples=$max_samples \
            --min_num_variables=3 \
            --Cp=0.1 \
            --root_dir=theory_logs \
            --seed=$seed
    done
done

func_list=(levy10_50 levy10_100 levy10_300)
for func in ${func_list[@]}
do
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        python3 mcts_vs_theory.py \
            --seed=$seed \
            --func=$func \
            --max_samples=$max_samples \
            --min_num_variables=5 \
            --Cp=10 \
            --root_dir=theory_logs \
            --seed=$seed
    done
done

# python3 average_theory.py