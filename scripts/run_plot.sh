#!/bin/bash

func_list=(hartmann6_300 hartmann6_500)
for func in ${func_list[@]}
do
    python3 plot.py --func=$func --root_dir=logs/hartmann6_logs/ --output_name=results/${func}.pdf
done
