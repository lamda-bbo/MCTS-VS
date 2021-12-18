import pandas as pd
import matplotlib.pyplot as plt
from plot import load_results


results = load_results('simple_logs', False)
all_result = dict()

for i in range(len(results)):
    result = results[i]
    name, data = result.name, result.progress
    name = name.strip('.csv')
    func, algo, seed = name.split('-')
    func = func + seed
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
    value_df_dict[func] = pd.DataFrame({'x': range(1000)})
    for algo in all_result[func].keys():
        value_df_dict[func] = pd.merge(value_df_dict[func], all_result[func][algo], on='x')
        value_df_dict[func] = value_df_dict[func].rename(columns={'y': algo})
    x = value_df_dict[func]['x']
    value_df_dict[func] = value_df_dict[func].drop('x', axis=1)
    rank_df_dict[func] = value_df_dict[func].rank(axis=1, ascending=False)
    rank_df_dict[func]['x'] = x

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

plt.figure(figsize=(16, 12))
key_map = {
    'bo': ('Vanilla BO', 'magenta'),
    'dropout3': ('Dropout3', 'yellow'),
    'dropout6': ('Dropout6', 'green'),
    'dropout10': ('Dropout10', 'red'),
    'dropout15': ('Dropout15', 'pink'),
    'lamcts_bo': ('LaMCTS-BO', 'purple'),
    'lamcts_vs_bo': ('LVS-BO', 'blue'),
}
for algo in key_map.keys():
    plt.plot(x, rank_mean[algo], label=key_map[algo][0], color=key_map[algo][1])
plt.legend(loc='best')
plt.xlabel('Evaluations')
plt.ylabel('Average rank')
plt.savefig('results/average_rank.pdf')
    