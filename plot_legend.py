import matplotlib.pyplot as plt
import argparse
from plot import color_map

parser = argparse.ArgumentParser()
parser.add_argument('--type', required=True, type=str)
parser.add_argument('--output_name', required=True, type=str)
args = parser.parse_args()

# 
# 
if args.type == 'exp1_1':
    key = ['Vanilla BO', 'Dropout-BO', 'LVS-BO']
elif args.type == 'exp1_2':
    key = ['TuRBO', 'Dropout-TuRBO', 'LVS-TuRBO']
elif args.type == 'exp2':
    key = ['HeSBO', 'ALEBO', 'LA-MCTS-TuRBO', 'CMA-ES', 'LVS-BO', 'LVS-TuRBO']
elif args.type == 'rl':
    key = ['HeSBO', 'LA-MCTS-TuRBO', 'CMA-ES', 'LVS-BO', 'LVS-TuRBO']
else:
    assert 0
colors = [color_map[k] for k in key]
labels = key
n = len(colors)

f = lambda m,c: plt.plot([], [],marker=m, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(n)]

legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False, ncol=100, bbox_to_anchor=(1,1))

fig = legend.figure
fig.canvas.draw()
bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(args.output_name + '_legend.pdf', dpi="figure", bbox_inches=bbox)

