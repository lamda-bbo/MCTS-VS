import matplotlib
params = {
    'lines.linewidth': 2,
    'legend.fontsize': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
}
matplotlib.rcParams.update(params)
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import collections

from plot_tool import plot_util as pu


COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue']

color_map = {
    '11': '11'
}

max_samples = 300

Result = collections.namedtuple('Result', 'name progress')


def load_results(root_dir, verbose=True):
    all_results = []
    for func_name in os.listdir(root_dir):
        if func_name.startswith('.'):
            continue
        for dirname in os.listdir(os.path.join(root_dir, func_name)):
            if dirname.endswith('.csv'):
                name = '%s-%s' % (func_name, dirname)
                progress = pd.read_csv(os.path.join(root_dir, func_name, dirname))
                # progress = progress[progress['x'] <= max_samples]
                # progress = progress[progress['y'] >= 0.92]
                # progress = progress[progress['t'] < 1200]
                result = Result(name=name, progress=progress)
                all_results.append(result)
                print('load %s ' % name)
    print('load %d results' % len(all_results))
    return all_results


def draw(xy_fn, split_fn, group_fn, xlabel, ylabel, max_x, interval_x):
    plt.figure(dpi=300)
    fig, axarr = pu.plot_results(all_results, xy_fn=xy_fn, split_fn=split_fn, group_fn=group_fn, shaded_std=True, shaded_err=False, average_group=True, tiling='horizontal', xlabel=xlabel, ylabel=ylabel)
    plt.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.2, left=0.08, top=0.95)
    for ax in axarr[0]:
        ax.set_xticks(np.arange(0, max_x, interval_x))
        ax.set_xticklabels([str(i) for i in np.arange(0, max_x, interval_x)])
    plt.savefig(args.output_name, bbox_inches='tight')

    
def xy_fn(r):
    return r.progress['x'], r.progress['y']

def ty_fn(r):
    return r.progress['t'], r.progress['y']
    
def split_fn(r):
    name = r.name
    splits = name.split('-')
    return splits[0]


def main(root_dir):
    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        if alg_name == 'bo':
            return 'Vanilla BO'
        elif alg_name.startswith('dropout'):
            # return 'Dropout-' + alg_name[7: ]
            return 'Dropout-BO'
        elif alg_name == 'lamcts_vs_bo':
            return 'LVS-BO'
        elif alg_name == 'lamcts_bo':
            return 'Lamcts-BO'
        elif alg_name == 'rembo':
            return 'REMBO'
        else:
            raise ValueError('%s not supported' % alg_name)
    
    draw(xy_fn, split_fn, group_fn, 'Evaluations', 'Function value', 600, 100)
    # draw(ty_fn, split_fn, group_fn, 'Time(sec)', 'Function value', 600, 100)
    
    
def cp_plot(root_dir):
    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        cp = alg_name.split('_')[-1]
        alg_name = alg_name[: 12]
        if alg_name == 'lamcts_vs_bo':
            return 'Lamcts-VS-BO(' + cp + ')'
        else:
            raise ValueError('%s not supported' % alg_name)
            
    assert 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, type=str)
    parser.add_argument('--output_name', required=True, type=str)
    args = parser.parse_args()
    
    all_results = load_results(args.root_dir, verbose=True)
    main(root_dir=args.root_dir)
    # cp_plot(root_dir=args.root_dir)
