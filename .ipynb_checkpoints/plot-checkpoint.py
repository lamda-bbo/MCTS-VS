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

import plot_util as pu

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


def plot_k(root_dir):
    all_results = load_results(root_dir, verbose=True)

    def xy_fn(r):
        x = np.cumsum(r.progress['xs'])
        y = r.progress['ys']
        return x, y

    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return splits[0]

    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        if alg_name == 'ES':
            return 'Vanilla ES'
        elif alg_name == 'SGES100':
            return 'SGES(k=100)'
        elif alg_name == 'SGES1':
            return 'SGES(k=1)'
        elif alg_name == 'SGES5':
            return 'SGES(k=5)'
        else:
            return 'SGES(k=%s)' % alg_name[-2:]

    plt.figure(dpi=300)
    fig, axarr = pu.plot_results(all_results, xy_fn=xy_fn, split_fn=split_fn, group_fn=group_fn, shaded_std=True,
                                 shaded_err=False, average_group=True, tiling='horizontal',
                                  xlabel='Evaluations', ylabel='Loss')
    plt.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.2, left=0.08, top=0.95)
    for ax in axarr[0]:
        ax.set_xticks(np.arange(0, 12.5e4, 2.5e4))
        ax.set_xticklabels(['0', '25k', '50k', '75k', '100k'])
    # fig.text(0.5, 0.05, s='# Evaluation', fontsize=18)
    # fig.text(0.04, 0.5, s='Loss', fontsize=18, rotation='vertical')
    plt.savefig('blackbox_k.pdf', bbox_inches='tight')


def main(root_dir):
    all_results = load_results(root_dir, verbose=True)

    def xy_fn(r):
        x = np.cumsum(r.progress['xs'])
        y = r.progress['ys']
        return x, y

    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return splits[0]

    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        if alg_name == 'SGES':
            return 'SGES'
        elif alg_name == 'CMA':
            return 'CMA-ES'
        elif alg_name == 'GES':
            return 'Guided ES'
        elif alg_name == 'ES':
            return 'Vanilla ES'
        elif alg_name == 'ASEBO':
            return 'ASEBO'
        else:
            raise ValueError('%s not supported' % alg_name)

    plt.figure(dpi=300)
    fig, axarr = pu.plot_results(all_results, xy_fn=xy_fn, split_fn=split_fn, group_fn=group_fn, shaded_std=True,
                                 shaded_err=False, average_group=True, tiling='horizontal',
                                  xlabel='Evaluations', ylabel='Loss')
    plt.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.2, left=0.08, top=0.95)
    for ax in axarr[0]:
        ax.set_xticks(np.arange(0, 12.5e4, 2.5e4))
        ax.set_xticklabels(['0', '25k', '50k', '75k', '100k'])
    plt.savefig('blackbox.pdf', bbox_inches='tight')


def plot_accuracy(root_dir):
    all_results = load_results(root_dir, verbose=True)

    def xy_fn(r):
        x = np.cumsum(r.progress['xs'])
        y = r.progress['errors']
        return x, y

    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return splits[0]

    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        if alg_name == 'SGES':
            return 'SGES'
        elif alg_name == 'CMA':
            return 'CMA-ES'
        elif alg_name == 'GES':
            return 'Guided ES'
        elif alg_name == 'ES':
            return 'Vanilla ES'
        elif alg_name == 'ASEBO':
            return 'ASEBO'
        else:
            raise ValueError('%s not supported' % alg_name)
    
    _all_results = []
    for result in all_results:
        if 'Sphere' in result.name:
            if 'CMA' in result.name:
                continue
            _all_results.append(result)
    all_results = _all_results

    plt.figure(dpi=300)
    fig, axarr = pu.plot_results(all_results, xy_fn=xy_fn, split_fn=split_fn, group_fn=group_fn, shaded_std=True,
                                 shaded_err=False, average_group=True, tiling='horizontal',
                                  xlabel='Evaluations', ylabel='Cosine Similarity')
    plt.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.2, left=0.08, top=0.95)
    for ax in axarr[0]:
        ax.set_xticks(np.arange(0, 12.5e4, 2.5e4))
        ax.set_xticklabels(['0', '25k', '50k', '75k', '100k'])
    plt.savefig('blackbox_accuracy.pdf', bbox_inches='tight')




if __name__ == '__main__':
    main(root_dir='logs')
    # plot_k(root_dir='tmp')
    # plot_accuracy(root_dir='logs')
