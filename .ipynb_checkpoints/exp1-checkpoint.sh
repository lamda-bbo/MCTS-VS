#!/bin/bash

python3 lamcts_vs.py --func=hartmann6_300 --Cp=0.1 --root_dir=exp1 --seed=2023
# python3 ax_embedding_bo.py --func=hartmann6_300 --max_samples=600 --active_dims=27 --strategy=rembo --root_dir=exp1 --seed=2023
# python3 ax_embedding_bo.py --func=hartmann6_300 --max_samples=600 --active_dims=27 --strategy=alebo --root_dir=exp1 --seed=2022