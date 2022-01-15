import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os


root_dir = 'theory_result'
max_samples = 600
for func_name in os.listdir(root_dir):
    if func_name.startswith('.') or not os.path.isdir(os.path.join(root_dir, func_name)):
        continue
    average_recall_list = []
    for algo_name in os.listdir(os.path.join(root_dir, func_name)):
        if algo_name.startswith('.'):
            continue
        path = os.path.join(root_dir, func_name, algo_name)
        for file_name in os.listdir(path):
            # print(path, file_name)
            if file_name.startswith('recall') and file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(path, file_name))
                recall = df['recall']
                pre_t = 0
                sum_recall = 0
                for i in recall.index:
                    dt = df.iloc[i]['t'] - pre_t
                    sum_recall += dt * df.iloc[i]['recall']
                    pre_t = df.iloc[i]['t']
                average_recall = sum_recall / pre_t
                average_recall_list.append(average_recall)
    assert len(average_recall_list) == 5
    print('Average recall of {}: {}'.format(func_name, np.mean(average_recall_list)))