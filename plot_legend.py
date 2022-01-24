import matplotlib.pyplot as plt
import argparse
from plot import color_map

parser = argparse.ArgumentParser()
parser.add_argument('--ncols', required=False, type=int, default=100)
parser.add_argument('--type', required=True, type=str)
parser.add_argument('--output_name', required=True, type=str)
args = parser.parse_args()

# 
# 
if args.type == 'exp1_1':
    key = ['Vanilla BO', 'Dropout-BO', 'MCTS-VS-BO']
elif args.type == 'exp1_2':
    key = ['TuRBO', 'Dropout-TuRBO', 'MCTS-VS-TuRBO']
elif args.type == 'exp2':
    key = ['TuRBO', 'LA-MCTS-TuRBO', 'HeSBO', 'ALEBO', 'CMA-ES', 'MCTS-VS-BO', 'MCTS-VS-TuRBO']
elif args.type == 'rl':
    key = ['TuRBO', 'LA-MCTS-TuRBO', 'HeSBO', 'CMA-ES', 'MCTS-VS-BO', 'MCTS-VS-TuRBO']
else:
    assert 0
colors = [color_map[k] for k in key]
labels = key
n = len(colors)

f = lambda m,c: plt.plot([], [],marker=m, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(n)]

legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False, ncol=args.ncols, bbox_to_anchor=(1,1))

fig = legend.figure
fig.canvas.draw()
bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(args.output_name + '_legend.pdf', bbox_inches=bbox)

