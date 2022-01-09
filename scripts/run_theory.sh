#!/bin/bash

seed_start=2021
seed_end=2025

# for ((seed=$seed_start; seed<=$seed_end; seed++))
# do
#     python3 lamcts_vs_f.py \
#         --seed=$seed \
#         --func=branin2_12 \
#         --min_num_variables=2 \
#         --Cp=0.1 \
#         --root_dir=theory_logs \
#         --seed=$seed
# done

for ((seed=$seed_start; seed<=$seed_end; seed++))
do
    python3 lamcts_vs_f.py \
        --seed=$seed \
        --func=hartmann6_300 \
        --max_samples=600 \
        --min_num_variables=3 \
        --Cp=0.1 \
        --root_dir=theory_logs \
        --seed=$seed
done

python3 average_theory.py