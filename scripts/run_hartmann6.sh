#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

seed_start=2021
seed_end=2021

func_list=(hartmann6_300 hartmann6_500)
max_samples=600
Cp=0.1
root_dir=hartmann6_logs

for func in ${func_list[@]}
do
    # mcts-vs-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 mcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=$Cp \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # mcts-vs-turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 mcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --turbo_max_evals=50 \
            --Cp=$Cp \
            --ipt_solver=turbo \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done

    # mcts-vs-rs
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 mcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=$Cp \
            --ipt_solver=rs \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done

    # mcts-vs-saasbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 mcts_vs.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=$Cp \
            --ipt_solver=saasbo \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done

    # random search
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/random_search.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # vanilla bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/vanilia_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        } 
    done
    
    # dropout-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/dropout.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=6 \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done

    # dropout-turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/dropout.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=6 \
            --ipt_solver=turbo \
            --root_dir=$root_dir \
            --seed=$seed
        } 
    done
    
    # dropout-rs
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/dropout.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=6 \
            --ipt_solver=rs \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done

    # lasso-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/lasso_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=6 \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # lasso-turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/lasso_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=6 \
            --ipt_solver=turbo \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done

    # lamcts-turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/lamcts.py \
            --func=$func \
            --max_samples=$max_samples \
            --Cp=$Cp \
            --solver_type=turbo \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done

    # saasbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/saasbo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # turbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/turbo.py \
            --func=$func \
            --max_samples=$max_samples \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # hesbo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/ax_embedding_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=6 \
            --strategy=hesbo \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # alebo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/ax_embedding_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --active_dims=6 \
            --strategy=alebo \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
    
    # cmaes
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/cmaes.py \
            --func=$func \
            --max_samples=$max_samples \
            --pop_size=20 \
            --sigma=0.8 \
            --root_dir=$root_dir \
            --seed=$seed
        } 
    done

    # vae-bo
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        {
        python3 baseline/vae_bo.py \
            --func=$func \
            --max_samples=$max_samples \
            --update_interval=30 \
            --active_dims=6 \
            --lr=0.001 \
            --root_dir=$root_dir \
            --seed=$seed
        }
    done
done
