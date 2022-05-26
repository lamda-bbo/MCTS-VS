import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from plot import color_map

parser = argparse.ArgumentParser()
parser.add_argument('--ncols', required=False, type=int, default=100)
parser.add_argument('--type', required=True, type=str)
parser.add_argument('--output_name', required=True, type=str)
args = parser.parse_args()

os.makedirs('results/legend', exist_ok=True)


# if args.type == 'exp1_1':
#     key = ['Vanilla BO', 'Dropout-BO', 'MCTS-VS-BO']
# elif args.type == 'exp1_2':
#     key = ['TuRBO', 'Dropout-TuRBO', 'MCTS-VS-TuRBO']
# elif args.type == 'exp2':
#     # key = ['TuRBO', 'LA-MCTS-TuRBO', 'HeSBO', 'ALEBO', 'CMA-ES', 'MCTS-VS-BO', 'MCTS-VS-TuRBO', 'VAE-BO']
#     key = ['MCTS-VS-BO', 'MCTS-VS-TuRBO', 'TuRBO', 'LA-MCTS-TuRBO', 'HeSBO', 'ALEBO', 'VAE-BO', 'CMA-ES']
# elif args.type == 'rl':
#     # key = ['TuRBO', 'LA-MCTS-TuRBO', 'HeSBO', 'CMA-ES', 'MCTS-VS-BO', 'MCTS-VS-TuRBO']
#     key = ['MCTS-VS-BO', 'MCTS-VS-TuRBO', 'TuRBO', 'LA-MCTS-TuRBO', 'HeSBO', 'CMA-ES']
# elif args.type == 'appendix_exp':
#     key = ['MCTS-VS-BO', 'MCTS-VS-TuRBO', 'TuRBO', 'LA-MCTS-TuRBO', 'HeSBO']
# else:
#     assert 0
    
# key = ['MCTS-VS-BO', 'MCTS-VS-TuRBO', 'MCTS-VS-RS', 'RS', 'LA-MCTS-TuRBO', 'TuRBO']
key = ['MCTS-VS-RS', 'MCTS-VS-BO', 'MCTS-VS-TuRBO']
colors = [color_map[k] for k in key]
labels = key
n = len(colors)

f = lambda m,c: plt.plot([], [],marker=m, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(n)]

legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False, ncol=args.ncols, bbox_to_anchor=(1,1), columnspacing=1)

fig = legend.figure
fig.canvas.draw()

# bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

expand=[-1, -1, 1, 1]
bbox = legend.get_window_extent()
bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

fig.savefig(args.output_name + '_legend.pdf', bbox_inches=bbox)



