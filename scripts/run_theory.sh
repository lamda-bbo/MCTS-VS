#!/bin/bash

seed_start=2021
seed_end=2025

for ((seed=$seed_start; seed<=$seed_end; seed++))
do
    python3 lamcts_vs_f.py --seed=$seed --func=levy10_50 --Cp=5
done

# for ((seed=$seed_start; seed<=$seed_end; seed++))
# do
#     python3 lamcts_vs_f.py --func=hartmann6_100
# done

python3 average_theory.py