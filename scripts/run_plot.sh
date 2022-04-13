#!/bin/bash

# ================ synthetic function ===============

# func_list=(hartmann6_100 hartmann6_300 hartmann6_500)
# func_list=(hartmann90_500)
# # func_list=(hartmann6_100 hartmann6_1000)
# for func in ${func_list[@]}
# do
#     # python3 plot.py --func=$func --root_dir=logs/hartmann6_logs/ --output_name=results/${func}_exp1_1.pdf
#     # python3 plot.py --func=$func --root_dir=logs/hartmann6_logs/ --output_name=results/${func}_exp1_2.pdf
#     python3 plot.py --func=$func --root_dir=logs/hartmann6_logs/ --output_name=results/${func}_exp2.pdf
# done

func_list=(hartmann90_500)
# func_list=(hartmann60_500)
for func in ${func_list[@]}
do
    python3 plot.py --func=$func --root_dir=saved_logs/hartmann6_logs/ --output_name=results/${func}_exp2.pdf
done

# func_list=(levy10_50 levy10_100 levy10_300)
# for func in ${func_list[@]}
# do
#     # python3 plot.py --func=$func --root_dir=logs/levy10_logs/ --output_name=results/${func}_exp1_1.pdf
#     # python3 plot.py --func=$func --root_dir=logs/levy10_logs/ --output_name=results/${func}_exp1_2.pdf
#     python3 plot.py --func=$func --root_dir=logs/levy10_logs/ --output_name=results/${func}_exp2.pdf
# done

# ================= nasbench =================
# python3 plot.py --func=nasbench --root_dir=logs/nasbench_logs/ --output_name=results/nas.pdf
# python3 plot.py --func=nasbench --root_dir=logs/nasbench_logs/ --output_name=results/nas_partial.pdf
# python3 plot.py --func=nasbench --root_dir=logs/nasbench_logs/ --output_name=results/nas_time.pdf
# python3 plot.py --func=nasbench --root_dir=logs/nasbench_logs/ --output_name=results/nas_time_partial.pdf

# ================= hpobench =================
# python3 plot.py --func=nasbench201 --root_dir=logs/hpobench_logs/ --output_name=results/nas201.pdf
# python3 plot.py --func=nasbench1shot1 --root_dir=logs/hpobench_logs/ --output_name=results/nas1shot1.pdf
# python3 plot.py --func=nasbench201 --root_dir=logs/hpobench_logs/ --output_name=results/nas201_time.pdf
# python3 plot.py --func=nasbench1shot1 --root_dir=logs/hpobench_logs/ --output_name=results/nas1shot1_time.pdf

# python3 plot.py --func=nasbench201 --root_dir=logs/naslib_logs/ --output_name=results/naslib_nas201_time.pdf
# python3 plot.py --func=nasbenchtrans --root_dir=logs/naslib_logs/ --output_name=results/naslib_nastrans_time.pdf

# python3 plot.py --func=nasbench201 --root_dir=logs/naslib_logs_parallel/ --output_name=results/naslib_nas201_time_parallel.pdf
# python3 plot.py --func=nasbenchtrans --root_dir=logs/naslib_logs_parallel/ --output_name=results/naslib_nastrans_time_parallel.pdf

# ================== rl =======================
# python3 plot.py --func=Hopper --root_dir=logs/rl_logs/ --output_name=results/hopper.pdf
# python3 plot.py --func=Walker2d --root_dir=logs/rl_logs/ --output_name=results/walker.pdf

# =============== ablation =====================
# python3 plot.py --func=strategy --legend_show=True --root_dir=logs/ablation_logs --output_name=results/ablation/ablation_strategy.pdf

# func_list=(hartmann6_300 hartmann6_500 levy10_100 levy10_300)
# for func in ${func_list[@]}
# do
#     python3 plot.py --func=${func}_Cp --legend_show=True --root_dir=logs/ablation_logs/ --output_name=results/ablation/${func}_Cp.pdf
# done

# python3 plot.py --func=min_num_variables --legend_show=True --root_dir=logs/ablation_logs --output_name=results/ablation/ablation_min_num_variables.pdf
# python3 plot.py --func=num_samples --legend_show=True --root_dir=logs/ablation_logs --output_name=results/ablation/ablation_num_samples.pdf
# python3 plot.py --func=param_k --legend_show=True --root_dir=logs/ablation_logs --output_name=results/ablation/ablation_param_k.pdf
# python3 plot.py --func=N_bad --legend_show=True --root_dir=logs/ablation_logs --output_name=results/ablation/ablation_N_bad.pdf
