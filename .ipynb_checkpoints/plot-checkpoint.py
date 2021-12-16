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
                result = Result(name=name, progress=progress)
                all_results.append(result)
                print('load %s ' % name)
    print('load %d results' % len(all_results))
    return all_results


def main(root_dir):
    all_results = load_results(root_dir, verbose=True)

    def xy_fn(r):
        x = r.progress['x']
        y = r.progress['y']
        return x, y

    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return splits[0]

    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        if alg_name == 'bo':
            return 'Vanilia BO'
        elif alg_name == 'dropout3':
            return 'Dropout-3'
        elif alg_name == 'dropout6':
            return 'Dropout-6'
        elif alg_name == 'dropout10':
            return 'Dropout-10'
        elif alg_name == 'dropout20':
            return 'Dropout-20'
        elif alg_name == 'dropout30':
            return 'Dropout-30'
        elif alg_name == 'lamcts_vs':
            return 'Lamcts-VS'
        elif alg_name == 'lamcts':
            return 'Lamcts'
        else:
            raise ValueError('%s not supported' % alg_name)

    plt.figure(dpi=300)
    fig, axarr = pu.plot_results(all_results, xy_fn=xy_fn, split_fn=split_fn, group_fn=group_fn, shaded_std=True,
                                 shaded_err=False, average_group=True, tiling='horizontal',
                                  xlabel='Evaluations', ylabel='Loss')
    plt.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.2, left=0.08, top=0.95)
    for ax in axarr[0]:
        ax.set_xticks(np.arange(0, 1000, 100))
        ax.set_xticklabels([str(i) for i in np.arange(0, 1000, 100)])
    plt.savefig(args.output_name, bbox_inches='tight')
    
    
def cp_plot(root_dir):
    all_results = load_results(root_dir, verbose=True)

    def xy_fn(r):
        x = r.progress['x']
        y = r.progress['y']
        return x, y

    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return splits[0]

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

    plt.figure(dpi=300)
    fig, axarr = pu.plot_results(all_results, xy_fn=xy_fn, split_fn=split_fn, group_fn=group_fn, shaded_std=True,
                                 shaded_err=False, average_group=True, tiling='horizontal',
                                  xlabel='Evaluations', ylabel='Loss')
    plt.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.2, left=0.08, top=0.95)
    for ax in axarr[0]:
        ax.set_xticks(np.arange(0, 1000, 100))
        ax.set_xticklabels([str(i) for i in np.arange(0, 1000, 100)])
    plt.savefig(args.output_name, bbox_inches='tight')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, type=str)
    parser.add_argument('--output_name', required=True, type=str)
    args = parser.parse_args()
    # main(root_dir=args.root_dir)
    cp_plot(root_dir=args.root_dir)
