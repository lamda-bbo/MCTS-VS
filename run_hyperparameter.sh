#!/bin/bash

max_samples=1000

func_list=('hartmann6' 'hartmann6_50' 'hartmann6_100' 'hartmann6_300' 'hartmann6_500')
Cp_list=(0.1 0.5 1 5 10)

for((func_idx=0; func_idx<${#func_list[@]}; func_idx++))
do
    func=${func_list[${func_idx}]}
    echo 'test function: ' ${func}
    for((cp_idx=0; cp_idx<${#Cp_list[@]}; cp_idx++))
    do
        for((i=42; i<=44; i++))
        do 
            {
            python3 lamcts_vs.py \
            --func=${func} \
            --max_samples=${max_samples} \
            --Cp=${Cp_list[${cp_idx}]} \
            --ipt_solver='bo' \
            --uipt_solver='bestk' \
            --root_dir='hyperparameter_logs' \
            --postfix="Cp${Cp_list[${cp_idx}]}" \
            --seed=${i}
            } &
        done
        wait
    done
done


func_list=('levy10' 'levy10_50' 'levy10_100' 'levy10_300' 'levy10_500' 'levy20' 'levy20_50' 'levy20_100' 'levy20_300' 'levy20_500')
Cp_list=(1 5 10 15 20)

for((func_idx=0; func_idx<${#func_list[@]}; func_idx++))
do
    func=${func_list[${func_idx}]}
    echo 'test function: ' ${func}
    for((cp_idx=0; cp_idx<${#Cp_list[@]}; cp_idx++))
    do
        for((i=42; i<=44; i++))
        do 
            {
            python3 lamcts_vs.py \
            --func=${func} \
            --max_samples=${max_samples} \
            --Cp=${Cp_list[${cp_idx}]} \
            --ipt_solver='bo' \
            --uipt_solver='bestk' \
            --root_dir='hyperparameter_logs' \
            --postfix="Cp${Cp_list[${cp_idx}]}" \
            --seed=${i}
            } &
        done
        wait
    done
done