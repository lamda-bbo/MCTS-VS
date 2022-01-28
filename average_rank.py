import pandas as pd
import matplotlib.pyplot as plt
import os
import collections

import matplotlib
params = {
    'lines.linewidth': 2,
    'legend.fontsize': 25,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
}
matplotlib.rcParams.update(params)

Result = collections.namedtuple('Result', 'name progress')

def load_results(root_dir, verbose=True):
    all_results = []
    for func_name in os.listdir(root_dir):
        if func_name.startswith('.'):
            continue
        print(func_name)
        # if func_name == 'hartmann6_500' or func_name == 'levy10_500' or func_name == 'levy20_500':
        #     continue
        if func_name.startswith('levy20'):
            continue
        
        for dirname in os.listdir(os.path.join(root_dir, func_name)):
            if not (dirname.startswith('mcts_vs') or dirname.startswith('dropout')):
                continue
            if dirname.endswith('.csv'):
                name = '%s-%s' % (func_name, dirname)
                progress = pd.read_csv(os.path.join(root_dir, func_name, dirname))
                result = Result(name=name, progress=progress)
                all_results.append(result)
                print('load %s ' % name)
    print('load %d results' % len(all_results))
    return all_results


results = load_results('saved_logs/dropout_logs', False)
all_result = dict()

for i in range(len(results)):
    result = results[i]
    name, data = result.name, result.progress
    name = name.strip('.csv')
    func, algo, seed = name.split('-')
    func = func + seed
    if algo.startswith('dropout'):
        algo = algo.split('_')
        algo = algo[0] + '_' + algo[2]
    
    if all_result.get(func, None) is None:
        all_result[func] = dict()
    if all_result[func].get(algo, None) is None:
        all_result[func][algo] = pd.DataFrame({'x': data['x'], 'y': data['y']})
    else:
        assert 0
        all_result[func][algo]['y'] += data['y']
value_df_dict = dict()
rank_df_dict = dict()

for func in all_result.keys():
    value_df_dict[func] = pd.DataFrame({'x': range(600)})
    for algo in all_result[func].keys():
        value_df_dict[func] = pd.merge(value_df_dict[func], all_result[func][algo], on='x')
        value_df_dict[func] = value_df_dict[func].rename(columns={'y': algo})
    x = value_df_dict[func]['x']
    value_df_dict[func] = value_df_dict[func].drop('x', axis=1)
    rank_df_dict[func] = value_df_dict[func].rank(axis=1, ascending=False)
    rank_df_dict[func]['x'] = x
# print(rank_df_dict)
rank_sum = dict()
rank_cnt = dict()
x = None

for func in rank_df_dict.keys():
    for algo in rank_df_dict[func].columns:
        if algo == 'x':
            x = rank_df_dict[func]['x']
            continue
        if rank_sum.get(algo, None) is None:
            rank_sum[algo] = rank_df_dict[func][algo]
        else:
            rank_sum[algo] += rank_df_dict[func][algo]
        rank_cnt[algo] = rank_cnt.get(algo, 0) + 1

rank_mean = dict()
for algo in rank_sum.keys():
    rank_mean[algo] = rank_sum[algo] / rank_cnt[algo]
    
print(rank_cnt)

plt.figure(figsize=(16, 12))
key_map = {
    # 'bo': ('Vanilla BO', 'magenta'),
    'mcts_vs_bo': ('MCTS-VS-BO', 'crimson'),
    'dropout_3': ('Dropout-3', (62, 122, 178)),
    'dropout_6': ('Dropout-6', (76, 175, 73)),
    'dropout_10': ('Dropout-10', (152, 78, 163)),
    'dropout_15': ('Dropout-15', (255, 176, 105)),
    'dropout_20': ('Dropout-20', (141, 84, 71)),
    'dropout_30': ('Dropout-30', (71, 71, 71)),
}
for k, v in key_map.items():
    if isinstance(v[1], tuple):
        key_map[k] = (v[0], tuple([i/255 for i in v[1]]))
for algo in key_map.keys():
    if algo in rank_mean.keys():
        plt.plot(x, rank_mean[algo], label=key_map[algo][0], color=key_map[algo][1])
        print('plot: {}'.format(algo))
plt.title(r'Dropout with different $d$')
plt.legend(loc='best')
plt.xlabel('Number of evaluations')
plt.ylabel('Average rank')
plt.savefig('results/average_rank.pdf')
    