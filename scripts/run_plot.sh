#!/bin/bash

# ================ synthetic function ===============

# func_list=(hartmann6_100 hartmann6_300 hartmann6_500 hartmann6_1000)
# for func in ${func_list[@]}
# do
#     # python3 plot.py --func=$func --root_dir=saved_logs/hartmann6_logs/ --output_name=results/${func}_exp1_1.pdf
#     # python3 plot.py --func=$func --root_dir=saved_logs/hartmann6_logs/ --output_name=results/${func}_exp1_2.pdf
#     python3 plot.py --func=$func --root_dir=saved_logs/hartmann6_logs/ --output_name=results/${func}_exp2.pdf
# done

# func_list=(levy10_50 levy10_100 levy10_300)
# for func in ${func_list[@]}
# do
#     # python3 plot.py --func=$func --root_dir=saved_logs/levy10_logs/ --output_name=results/${func}_exp1_1.pdf
#     # python3 plot.py --func=$func --root_dir=saved_logs/levy10_logs/ --output_name=results/${func}_exp1_2.pdf
#     python3 plot.py --func=$func --root_dir=saved_logs/levy10_logs/ --output_name=results/${func}_exp2.pdf
# done

# ================= nasbench =================
# python3 plot.py --func=nasbench --root_dir=saved_logs/nasbench_logs/ --output_name=results/nas.pdf
# python3 plot.py --func=nasbench --root_dir=saved_logs/nasbench_logs/ --output_name=results/nas_partial.pdf
# python3 plot.py --func=nasbench --root_dir=saved_logs/nasbench_logs/ --output_name=results/nas_time.pdf
python3 plot.py --func=nasbench --root_dir=saved_logs/nasbench_logs/ --output_name=results/nas_time_partial.pdf

# ================== rl =======================
# python3 plot.py --func=Hopper --root_dir=saved_logs/rl_logs/ --output_name=results/hopper.pdf
# python3 plot.py --func=Walker2d --root_dir=saved_logs/rl_logs/ --output_name=results/walker.pdf

# =============== ablation =====================
# python3 plot.py --func=strategy --legend_show=True --root_dir=saved_logs/ablation_logs --output_name=results/ablation/ablation_strategy.pdf

# func_list=(hartmann6_300 hartmann6_500 levy10_100 levy10_300)
# for func in ${func_list[@]}
# do
#     python3 plot.py --func=${func}_Cp --legend_show=True --root_dir=saved_logs/ablation_logs/ --output_name=results/ablation/${func}_Cp.pdf
# done

# python3 plot.py --func=min_num_variables --legend_show=True --root_dir=saved_logs/ablation_logs --output_name=results/ablation/ablation_min_num_variables.pdf
# python3 plot.py --func=num_samples --legend_show=True --root_dir=saved_logs/ablation_logs --output_name=results/ablation/ablation_num_samples.pdf
# python3 plot.py --func=param_k --legend_show=True --root_dir=saved_logs/ablation_logs --output_name=results/ablation/ablation_param_k.pdf
