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

key_map = {
    # combined with vanilla bo
    'lamcts_vs_bo': 'LVS-BO',
    'bo': 'Vanilla BO',
    'lamcts_bo': 'LA-MCTS-BO',
    'rembo': 'REMBO',
    
    # conbined with turbo
    'lamcts_vs_turbo': 'LVS-TuRBO',
    'turbo1': 'TuRBO',
    'lamcts_turbo': 'LA-MCTS-TuRBO',
    'alebo': 'ALEBO',
    'hesbo': 'HeSBO',
    'cmaes': 'CMA-ES',
}

color_map = {
    # combined with vanilla bo
    'LVS-BO': 'crimson',#
    'Vanilla BO': 'gray',#
    'Dropout-BO': 'darkorange',#
    'LA-MCTS-BO': 'royalblue',
    'REMBO': 'magenta',
    
    # conbined with turbo
    'LVS-TuRBO': 'cyan',#
    'TuRBO': 'green',#
    'Dropout-TuRBO': 'brown',
    'LA-MCTS-TuRBO': 'royalblue',
    'ALEBO': 'brown',
    'HeSBO': 'olive',
    'CMA-ES': 'blueviolet',
    
    # ============== ablation =============
    # 
    'bestk': 'crimson',
    'random': 'royalblue',
    'copy': 'darkgreen',
    'mix': 'lavender',
    
    # Cp
    'Cp=0.01': 'crimson',
    'Cp=0.1': 'royalblue',
    'Cp=1': 'darkgreen',
    
    # min_num_variables
    'd=3': 'crimson',
    'd=6': 'royalblue',
    'd=10': 'darkgreen',
    'd=20': 'lavender',
    'd=50': 'salmon',
    
    '$N_{feature}$=2,$N_{sample}$=3': 'crimson',
    '$N_{feature}$=2,$N_{sample}$=5': 'royalblue',
    '$N_{feature}$=2,$N_{sample}$=10': 'darkgreen',
    '$N_{feature}$=5,$N_{sample}$=3': 'lavender',
    '$N_{feature}$=5,$N_{sample}$=5': 'salmon',
    '$N_{feature}$=5,$N_{sample}$=10': 'black',
}


exp1_algo_1 = (
    'lamcts_vs_bo',
    'dropout_bo',
    'bo',
)

exp1_algo_2 = (
    'lamcts_vs_turbo',
    'dropout_turbo',
    'turbo',
)

exp2_algo = (
    'lamcts_vs_bo',
    'lamcts_vs_turbo',
    'turbo',
    'hesbo',
    'alebo',
    'cmaes',
    'lamcts',
    'bo',
)


Result = collections.namedtuple('Result', 'name progress')


def load_results(root_dir, verbose=True):
    all_results = []
    for func_name in os.listdir(root_dir):
        if func_name.startswith('.'):
            continue
            
        if not func_name.startswith(args.func_name):
            continue
        
        for dirname in os.listdir(os.path.join(root_dir, func_name)):
            if dirname.startswith('rembo'):
                continue
                
            # if not dirname.startswith(exp1_algo_1):
            # if not dirname.startswith(exp1_algo_2):
            if not dirname.startswith(exp2_algo):
                continue
                
            if dirname.endswith('.csv'):
                name = '%s-%s' % (func_name, dirname)
                progress = pd.read_csv(os.path.join(root_dir, func_name, dirname))
                
                if func_name.startswith('levy10') or func_name.startswith('levy20'):
                    progress = progress[progress['y'] >= -50]
                if func_name.startswith('Hopper') or func_name.startswith('Walker'):
                    progress = progress[progress['x'] <= 2000]
                if func_name.startswith('nas'):
                    progress.loc[(progress['y'] < 0.90), 'y'] = 0.90
                    # progress = progress[progress['x'] <= 100]
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
    print('save to {}'.format(args.output_name))

    
def xy_fn(r):
    return r.progress['x'], r.progress['y']

def ty_fn(r):
    return r.progress['t'], r.progress['y']


def main(root_dir):
    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return splits[0].title()
    
    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        if alg_name.startswith('dropout'):
            _, solver_type, d = alg_name.split('_')
            if solver_type == 'bo':
                return 'Dropout-BO'
            elif solver_type == 'turbo':
                return 'Dropout-TuRBO'
            else:
                assert 0
        else:
            return key_map[alg_name]
    
    # synthetic function
    # draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Value', 600, 100)
    
    # nasbench
    # draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Value', 200, 50)
    # draw(ty_fn, split_fn, group_fn, 'Time(sec)', 'Value', 4000, 1000)
    
    # rover 
    # draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Value', 1500, 300)
    # draw(ty_fn, split_fn, group_fn, 'Time(sec)', 'Value', 4000, 1000)
    
    # rl
    draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Reward', 2000, 500)
    
    
def ablation_strategy(root_dir):
    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return 'Fill-in strategy'
    
    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        
        return alg_name.split('_')[-1]
    
    draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Value', 600, 100)
    

def ablation_Cp(root_dir):
    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return 'Cp'
    
    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        
        return 'Cp='+alg_name.split('_')[-1]
    
    draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Value', 600, 100)
    
    
def ablation_min_num_variables(root_dir):
    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return 'min_num_variables'
    
    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        
        return 'd='+alg_name.split('_')[-1]
    
    draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Value', 600, 100)
    
    
def ablation_num_samples(root_dir):
    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return 'Number of samples'
    
    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        f_bs, s_bs = alg_name.split('_')[-2], alg_name.split('_')[-1]
        
        return r'$N_{feature}$=' + f_bs + r',$N_{sample}$=' + s_bs
    
    draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Value', 600, 100)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--func_name', required=True, type=str)
    parser.add_argument('--root_dir', required=True, type=str)
    parser.add_argument('--output_name', required=True, type=str)
    args = parser.parse_args()
    
    all_results = load_results(args.root_dir, verbose=True)
    
    # main(root_dir=args.root_dir)
    ablation_strategy(root_dir=args.root_dir)
    # ablation_Cp(root_dir=args.root_dir)
    # ablation_min_num_variables(root_dir=args.root_dir)
    # ablation_num_samples(root_dir=args.root_dir)
