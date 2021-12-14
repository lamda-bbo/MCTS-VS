#!/bin/bash

func_list=('hartmann6' 'hartmann6_50' 'hartmann6_100' 'hartmann6_300' 'hartmann6_500' 'levy10' 'levy10_50' 'levy10_100' 'levy10_300' 'levy10_500' 'levy20' 'levy20_50' 'levy20_100' 'levy20_300' 'levy20_500')
# func=${func_list[2]}

for((func_idx=0; func_idx<${#func_list[@]}; func_idx++))
do
    func=${func_list[${func_idx}]}
    echo 'test function: ' ${func}

    for((i=42; i<=44; i++))
    do 
        {
        python3 lamcts_vs.py \
        --func=${func} \
        --max_samples=1000 \
        --seed=${i}
        } &
    done
    wait

    for((i=42; i<=44; i++))
    do
        {
        python3 vanilia_bo.py \
        --func=${func} \
        --max_samples=1000 \
        --init_samples=10 \
        --batch_size=3 \
        --seed=${i} 
        } &
    done
    wait

    active_dims=(3 6 10 20 30)

    for ((idx=0; idx<${#active_dims[@]}; idx++))
    do
        for ((i=42; i<=44; i++))
        do 
            {
            python3 dropout.py \
            --func=${func} \
            --max_samples=1000 \
            --init_samples=10 \
            --batch_size=3 \
            --active_dims=${active_dims[${idx}]} \
            --seed=${i}
            } &
        done
        wait
    done
done
