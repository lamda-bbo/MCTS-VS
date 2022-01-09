import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os


# read .csv files
root_dir = 'theory_result'
max_samples = 600
recall_list = []
precision_list = []
n_selected_list = []
for file_name in os.listdir(root_dir):
    path = os.path.join(root_dir, file_name)
    if (file_name.startswith('recall') or file_name.startswith('precision')) and file_name.endswith('.csv'):
        if file_name.startswith('recall'):
            file_type = 'recall'
        else:
            file_type = 'precision'
        
        # interpolation
        file = pd.read_csv(path)
        tmp = []
        for i in range(max_samples):
            if len(file.loc[file['t'] < i]) > 0:
                tmp.append((i, file.loc[file['t'] < i].iloc[-1][file_type]))
            else:
                tmp.append((i, 0))
        
        # update
        if file_type == 'recall':
            recall_list.append(pd.DataFrame(tmp, columns=['t', 'recall']))
        else:
            precision_list.append(pd.DataFrame(tmp, columns=['t', 'precision']))
    
    elif file_name.startswith('n_selected'):
        file = pd.read_csv(path)
        tmp = []
        for i in range(max_samples):
            if len(file.loc[file['t'] < i]) > 0:
                tmp.append((i, file.loc[file['t'] < i].iloc[-1]['n']))
            else:
                tmp.append((i, 0))
        n_selected_list.append(pd.DataFrame(tmp, columns=['t', 'n']))
    else:
        continue

# calculate the sum
sum_recall = 0
for recall in recall_list:
    sum_recall += recall['recall']
average_recall = sum_recall / len(recall_list)

print('recall mean:', np.mean(average_recall))

sum_precision = 0
for precision in precision_list:
    sum_precision += precision['precision']
average_precision = sum_precision / len(precision_list)

sum_n_selected = 0
for n_selected in n_selected_list:
    sum_n_selected += n_selected['n']
average_n_selected = sum_n_selected / len(n_selected_list)
    
average_random = average_n_selected / 300

# print(average_recall)
# print(average_precision)

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(average_n_selected, color='b', label='#selected')
# ax1.legend(loc=1)
# ax2 = ax1.twinx()
# ax2.plot(average_recall, color='r', label='recall')
# ax2.plot(average_random, color='g', label='random')
# ax2.legend(loc=2)
# plt.savefig('theory_result/twinx.png')

fig = plt.figure()
plt.plot(average_recall, color='r', label='recall')
plt.plot(average_random, color='g', label='random')
plt.legend(loc=1)
plt.savefig('theory_result/twinx.png')
print('save to theory_result/twinx.png')

# =====================================
# plt.figure()
# plt.plot(average_recall)
# plt.title('recall')
# plt.savefig('theory_result/average_recall.png')

# plt.figure()
# plt.plot(average_precision)
# plt.title('precision')
# plt.savefig('theory_result/average_precision.png')

# plt.figure()
# plt.plot(average_n_selected)
# plt.title('n_selected')
# plt.savefig('theory_result/average_n_selected.png')