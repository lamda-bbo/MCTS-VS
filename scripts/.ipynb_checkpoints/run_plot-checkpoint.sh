#!/bin/bash

# =============================================================
# python3 plot.py --root_dir=saved_logs/hartmann6_logs/ --output_name=results/saved_hartmann6.pdf
# python3 plot.py --root_dir=saved_logs/levy10_logs/ --output_name=results/saved_levy10.pdf
# python3 plot.py --root_dir=saved_logs/levy20_logs/ --output_name=results/saved_levy20.pdf




# func_list=(hartmann6_100 hartmann6_300 hartmann6_500)
# for func in ${func_list[@]}
# do
#     python3 plot.py --func=$func --root_dir=saved_logs/hartmann6_logs/ --output_name=results/${func}_exp1_1.pdf
#     # python3 plot.py --func=$func --root_dir=saved_logs/hartmann6_logs/ --output_name=results/${func}_exp1_2.pdf
#     # python3 plot.py --func=$func --root_dir=saved_logs/hartmann6_logs/ --output_name=results/${func}_exp2.pdf
# done

# func_list=(levy10_50 levy10_100 levy10_300)
# for func in ${func_list[@]}
# do
#     python3 plot.py --func=$func --root_dir=saved_logs/levy10_logs/ --output_name=results/${func}_exp1_1.pdf
#     # python3 plot.py --func=$func --root_dir=saved_logs/levy10_logs/ --output_name=results/${func}_exp1_2.pdf
#     # python3 plot.py --func=$func --root_dir=saved_logs/levy10_logs/ --output_name=results/${func}_exp2.pdf
# done

# ================= nasbench =================
# python3 plot.py --func=nasbench --root_dir=logs/nasbench_logs/ --output_name=results/nas.pdf
# python3 plot.py --func=nasbench --root_dir=logs/nasbench_logs/ --output_name=results/nas_time.pdf

# ==================== rover ========================
# python3 plot.py --func=rover --root_dir=logs/rover_logs/ --output_name=results/rover.pdf
# python3 plot.py --func=rover --root_dir=logs/rover_logs/ --output_name=results/rover_time.pdf

# ================== rl =======================
# python3 plot.py --func=Hopper --root_dir=saved_logs/rl_logs/ --output_name=results/hopper.pdf
# python3 plot.py --func=Walker --root_dir=saved_logs/rl_logs/ --output_name=results/walker.pdf

# =============== ablation =====================
python3 plot.py --func=strategy --root_dir=logs/ablation_logs --output_name=results/ablation_strategy.pdf
# python3 plot.py --func=Cp --root_dir=logs/ablation_logs --output_name=results/ablation_Cp.pdf
# python3 plot.py --func=min_num_variables --root_dir=logs/ablation_logs --output_name=results/ablation_min_num_variables.pdf
# python3 plot.py --func=num_samples --root_dir=logs/ablation_logs --output_name=results/ablation_num_samples.pdf
