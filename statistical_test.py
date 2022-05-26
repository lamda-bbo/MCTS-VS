import numpy as np
import pandas as pd
from scipy.stats import ranksums
import os
from collections import defaultdict

# func = 'hartmann6_300'
# func = 'hartmann6_500'
# func = 'levy10_100'
func = 'levy10_100'
root_dir = 'saved_logs/repeat_logs/{}'.format(func)
y = defaultdict(list)
y5 = defaultdict(list)

for file_name in os.listdir(root_dir):
    if file_name.startswith('.'):
        continue
    algo = file_name.split('-')[0]
    seed = file_name.split('-')[1].strip('.csv')
    progress = pd.read_csv(os.path.join(root_dir, file_name))
    if len(progress) < 600:
        print(file_name)
        assert 0
        continue
    y[algo].append(progress.iloc[599]['y'])
    if int(seed) >= 2021 and int(seed) <= 2025:
        y5[algo].append(progress.iloc[599]['y'])

alternative = 'two-sided'
for key in y.keys():
    result1 = ranksums(y['mcts_vs_bo'], y[key], alternative=alternative)
    # result2 = ranksums(y5['mcts_vs_bo'], y5[key], alternative='greater')
    result2 = ranksums(y['mcts_vs_bo'][: 5], y[key][: 5], alternative=alternative)
    print(len(y[key]))
    print('algo: {}, result: {}'.format(key, result1))
    # print('algo: {}, result: {}'.format(key, result2))
    print('mean: {}, std: {}\n'.format(np.mean(y[key]), np.std(y[key])))

print(y)
for k, v in y.items():
    print(k)
    print(len(v))
df = pd.DataFrame(y)
df.to_csv('{}_ranksum_data.csv'.format(func))