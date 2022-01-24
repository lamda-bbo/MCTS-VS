#!/bin/bash

python3 plot_legend.py --type=exp1_1 --output_name=results/exp1_1
python3 plot_legend.py --type=exp1_2 --output_name=results/exp1_2
python3 plot_legend.py --type=exp2 --output_name=results/exp2
python3 plot_legend.py --ncols=4 --type=exp2 --output_name=results/exp3
python3 plot_legend.py --type=rl --output_name=results/rl
python3 plot_legend.py --ncols=3 --type=rl --output_name=results/rl1
