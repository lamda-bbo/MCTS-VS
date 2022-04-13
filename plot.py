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
    'lamcts_vs_bo': 'MCTS-VS-BO',
    'mcts_vs_bo': 'MCTS-VS-BO',
    'bo': 'Vanilla BO',
    # 'lamcts_bo': 'LA-MCTS-BO',
    # 'rembo': 'REMBO',
    # 'vae': 'VAE-BO',
    
    'mcts_vs_rs': 'MCTS-VS-RS',
    'random_search': 'RS',
    
    # conbined with turbo
    'lamcts_vs_turbo': 'MCTS-VS-TuRBO',
    'mcts_vs_turbo': 'MCTS-VS-TuRBO',
    'turbo1': 'TuRBO',
    'lamcts_turbo': 'LA-MCTS-TuRBO',
    
    # other sota
    'alebo': 'ALEBO',
    'hesbo': 'HeSBO',
    'cmaes': 'CMA-ES',
}

color_map = {
    # combined with vanilla bo
    'MCTS-VS-BO': 'crimson',
    'Vanilla BO': 'gray',
    'Dropout-BO': 'darkorange',
    # 'LA-MCTS-BO': 'royalblue',
    # 'REMBO': 'magenta',
    'VAE-BO': (216, 207, 22),
    
    # conbined with turbo
    'MCTS-VS-TuRBO': 'royalblue',
    'TuRBO': (171, 197, 231),
    # 'Dropout-TuRBO': (30, 37, 74),
    'Dropout-TuRBO': 'blueviolet',
    'LA-MCTS-TuRBO': (0, 190, 190),
    
    'MCTS-VS-RS': 'orange',
    'Dropout-RS': 'yellow',
    'RS': 'yellow',
    
    # other sota
    'ALEBO': (141, 84, 71),
    # 'HeSBO': (104, 113, 5),
    'HeSBO': (124, 136, 6),
    'CMA-ES': (71, 71, 71),
    
    # ============== ablation =============
    # 
    'bestk': 'crimson',
    'random': (255, 176, 105),
    # 'copy': 'darkgreen',
    # 'mix': 'orange',
    
    # Cp
    'Cp=0.01': (62, 122, 178),
    'Cp=0.1': 'crimson',
    'Cp=1': (76, 175, 73),
    'Cp=10': (152, 78, 163),
    'Cp=100': (255, 176, 105),
    
    # min_num_variables
    '$N_{split}$=3': 'crimson',
    '$N_{split}$=6': (62, 122, 178),
    '$N_{split}$=10': (76, 175, 73),
    '$N_{split}$=20': (152, 78, 163),
    '$N_{split}$=50': (255, 176, 105),
    
    '$N_v$=2,$N_s$=3': 'crimson',
    '$N_v$=2,$N_s$=5': (62, 122, 178),
    '$N_v$=2,$N_s$=10': (76, 175, 73),
    '$N_v$=5,$N_s$=3': (255, 176, 105),
    '$N_v$=5,$N_s$=5': (152, 78, 163),
    '$N_v$=5,$N_s$=10': (71, 71, 71),
    
    '$k$=1': (62, 122, 178),
    '$k$=5': (76, 175, 73),
    '$k$=10': (152, 78, 163),
    '$k$=15': (255, 176, 105),
    '$k$=20': 'crimson',
    
    '$N_{bad}$=1': (62, 122, 178),
    '$N_{bad}$=5': 'crimson',
    '$N_{bad}$=10': (76, 175, 73),
    '$N_{bad}$=15': (152, 78, 163),
    '$N_{bad}$=20': (255, 176, 105),
}

for k, v in color_map.items():
    if isinstance(v, tuple):
        color_map[k] = tuple([i/255 for i in v])


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
    'mcts_vs_bo',
    'lamcts_vs_turbo',
    'mcts_vs_turbo',
    'mcts_vs_rs',
    'dropout_rs',
    'random_search',
    'turbo',
    'hesbo',
    'alebo',
    'cmaes',
    'lamcts',
    'vae',
)


Result = collections.namedtuple('Result', 'name progress')


def load_results(root_dir, verbose=True):
    all_results = []
    for func_name in os.listdir(root_dir):
        if func_name.startswith('.'):
            continue
            
        if func_name != args.func_name:
            continue
        
        for dirname in os.listdir(os.path.join(root_dir, func_name)):
            if dirname.startswith('rembo') or dirname.startswith('lamcts_bo'):
                continue
            if func_name.startswith('nas') and dirname.startswith('hesbo'):
                continue
                
            # if not dirname.startswith(exp1_algo_1):
            # if not dirname.startswith(exp1_algo_2):
            if not dirname.startswith(exp2_algo):
                continue
                
            if dirname.endswith('.csv'):
                name = '%s-%s' % (func_name, dirname)
                progress = pd.read_csv(os.path.join(root_dir, func_name, dirname))
                
                if func_name.startswith('levy10') or func_name.startswith('levy20'):
                    progress = progress[progress['y'] >= -60]
                if func_name.startswith('Hopper') or func_name.startswith('Walker'):
                    progress = progress[progress['x'] <= 1200]
                if func_name.startswith('nas'):
                    # progress.loc[(progress['y'] < 0.94), 'y'] = 0.94
                    # progress = progress[progress['y'] >= 0.90]
                    # progress = progress[progress['x'] >= 150]
                    progress = progress[progress['t'] <= 1000]
                    # pass
                result = Result(name=name, progress=progress)
                all_results.append(result)
                print('load %s ' % name)
    print('load %d results' % len(all_results))
    return all_results


def draw(xy_fn, split_fn, group_fn, xlabel, ylabel, max_x, interval_x):
    plt.figure(dpi=300)
    fig, axarr = pu.plot_results(all_results, xy_fn=xy_fn, split_fn=split_fn, group_fn=group_fn, shaded_std=True, shaded_err=False, average_group=True, tiling='horizontal', xlabel=xlabel, ylabel=ylabel, legend_show=args.legend_show)
    # fig, axarr = pu.plot_results(all_results, xy_fn=xy_fn, split_fn=split_fn, group_fn=group_fn, shaded_std=True, shaded_err=False, average_group=True, tiling='horizontal', xlabel=xlabel, ylabel=ylabel, legend_show=args.legend_show, resample=8)
    # plt.plot([0, 1000], [0.7349, 0.7349], c='gray', linestyle='--')
    # plt.axhline(0.7349, c='gray', linestyle='--')
    # plt.axhline(0.5738, c='gray', linestyle='--')
    # plt.axhline(0.55, c='gray', linestyle='--')
    plt.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.2, left=0.08, top=0.95)
    for ax in axarr[0]:
        ax.set_xticks(np.arange(0, max_x, interval_x))
        ax.set_xticklabels([str(i) for i in np.arange(0, max_x, interval_x)])
        
        # nas partial
        # ax.set_xticks(np.arange(150, 200, 10))
        # ax.set_xticklabels([str(i) for i in np.arange(150, 200, 10)])
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
        if splits[0] == 'nasbench':
            return 'NAS-Bench-101'
        elif splits[0] == 'nasbench201':
            return 'NAS-Bench-201'
        elif splits[0] == 'nasbench1shot1':
            return 'NAS-Bench-1shot1'
        elif splits[0] == 'nasbenchtrans':
            return 'TransNAS-Bench-101'
        elif splits[0] == 'hartmann60_500':
            return 'Hartmann6_10_500'
        elif splits[0] == 'hartmann90_500':
            return 'Hartmann6_15_500'
        else:
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
            elif solver_type == 'rs':
                return 'Dropout-RS'
            else:
                assert 0
        elif alg_name.startswith('vae'):
            return 'VAE-BO'
        else:
            return key_map[alg_name]
    
    # synthetic function
    draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Value', 600, 100)
    
    # nasbench
    # draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Accuracy', 200, 50)
    # draw(ty_fn, split_fn, group_fn, 'Time (sec)', 'Val accuracy', 1000, 200)
    # draw(ty_fn, split_fn, group_fn, 'Time (sec)', 'Accuracy', 100, 20)
    
    # rl
    # draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Reward', 1200, 300)
    
    
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
        return splits[0][: -3].title() + ' Cp'
    
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
        return r'$N_{split}$'
    
    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        
        return r'$N_{split}$='+alg_name.split('_')[-1]
    
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
        
        return r'$N_v$=' + f_bs + r',$N_s$=' + s_bs
    
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
        
        return r'$N_v$=' + f_bs + r',$N_s$=' + s_bs
    
    draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Value', 600, 100)
    
    
def ablation_param_k(root_dir):
    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return 'Parameter $k$ of best-$k$'
    
    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        k = alg_name.split('_')[-1]
        
        return r'$k$=' + k
    
    draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Value', 600, 100)
    
    
def ablation_N_bad(root_dir):
    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return r'$N_{bad}$'
    
    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        
        return r'$N_{bad}$='+alg_name.split('_')[-1]
    
    draw(xy_fn, split_fn, group_fn, 'Number of evaluations', 'Value', 600, 100)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--func_name', required=True, type=str)
    parser.add_argument('--legend_show', default=True, type=bool)
    parser.add_argument('--root_dir', required=True, type=str)
    parser.add_argument('--output_name', required=True, type=str)
    args = parser.parse_args()
    
    all_results = load_results(args.root_dir, verbose=True)
    
    os.makedirs('results', exist_ok=True)
    main(root_dir=args.root_dir)
    # ablation_strategy(root_dir=args.root_dir)
    # ablation_Cp(root_dir=args.root_dir)
    # ablation_min_num_variables(root_dir=args.root_dir)
    # ablation_num_samples(root_dir=args.root_dir)
    # ablation_param_k(root_dir=args.root_dir)
    # ablation_N_bad(root_dir=args.root_dir)
