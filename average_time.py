import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from plot import key_map


def get_key(file_name):
    for k, v in key_map.items():
        if file_name.startswith(k):
            return v
    if file_name.startswith('dropout'):
        _, solver_type, d = file_name.split('_')
        if solver_type == 'bo':
            return 'Dropout-BO'
        elif solver_type == 'turbo':
            return 'Dropout-TuRBO'
    assert 0, 'Error key: {}'.format(file_name)


# read .csv files
# root_dir = 'saved_logs/time_logs'
root_dir = 'logs/time_logs'
max_samples = 600
for func_name in os.listdir(root_dir):
    if func_name.startswith('.'):
        continue
    
    epoch_time_dict = dict()
    for file_name in os.listdir(os.path.join(root_dir, func_name)):
        if file_name.startswith('.'):
            continue
        k = get_key(file_name)
        path = os.path.join(root_dir, func_name, file_name)
        df = pd.read_csv(path)
        df = df[df['x'] <= 100]
        # epoch_time = df['t'].iloc[-1] / df['x'].iloc[-1]
        epoch_time = df['t'].iloc[-1]
        if epoch_time_dict.get(k, None) is None:
            epoch_time_dict[k] = [epoch_time]
        else:
            epoch_time_dict[k].append(epoch_time)
    
    average_epoch_time = dict()
    for k, v in epoch_time_dict.items():
        average_epoch_time[k] = np.mean(v)
    
    print('================================')
    print('Func: {}'.format(func_name))
    for k, v in average_epoch_time.items():
        print(k, v)
    print('================================')
