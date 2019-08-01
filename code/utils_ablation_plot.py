import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, os.path

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='50', help='50 or gtea')
parser.add_argument('--pooling', type=str, help='gaussian or binary')
args = parser.parse_args()

# dataset = ['50', 'gtea'][1]
# pooling=['RPGaussian', 'RPBinary'][2]

data_dir = '/is/ps2/yzhang/workspaces/TemporalActionParsing-FineGrained/code/RP_results'

pooling = 'RPBinary' if args.pooling == 'binary' else 'RPGaussian'
dataset = '50' if args.dataset == '50' else 'gtea'
dataset_fullname = '50Salads' if args.dataset == '50' else 'GTEA'

## the original work
data_list = []
for dim in [1,2,4,8]:
    for comp in [1,2,4,8,16]:
        filename = os.path.join(data_dir, 'result_{}{}_dim{}_comp{}.txt'.format(dataset, pooling, dim, comp))
        data = np.loadtxt(filename, delimiter=',')
        run_info = np.repeat(np.array([[int(dim), int(comp)]]), 6, axis=0)
        data_list.append(np.concatenate([data, run_info], axis=1))

data_all = np.concatenate(data_list, axis=0)
columns=['accuracy','edit score', 'f1 score', 'dimension', 'rank']
df = pd.DataFrame(data_all, columns=columns)

## the results with stronger regularization (higher droput ratio + more iterations)
data_list_drop = []
for dim in [2,4,8]:
    for comp in [1,2,4,8,16]:
        
        if dim > 2:
            #filename = 'test_result_rebuttal_{}_{}_dim{}_comp{}_drp0.4_ep500.txt'.format(dataset_fullname, pooling, dim, comp)
            filename = 'test_result_rebuttal_{}_{}_dim{}_comp{}_dr0.5_ep500.txt'.format(dataset_fullname, pooling, dim, comp)
        else:
            filename = os.path.join(data_dir, 'result_{}{}_dim{}_comp{}.txt'.format(dataset, pooling, dim, comp))


        data = np.loadtxt(filename, delimiter=',')
        # data = np.expand_dims(data, axis=0)
        run_info = np.repeat(np.array([[int(dim), int(comp)]]), 6, axis=0)
        # run_info = np.array([[int(dim), int(comp)]])
        data_list_drop.append(np.concatenate([data, run_info], axis=1))

data_all_drop = np.concatenate(data_list_drop, axis=0)
columns=['accuracy','edit score', 'f1 score', 'dimension', 'rank']
df_d = pd.DataFrame(data_all_drop, columns=columns)



sns.set(style="whitegrid")


plt.figure()
ax = sns.lineplot(x="dimension", y="accuracy", #hue="rank",
                  marker="o", 
                  legend=False, #err_style=None,
                  palette=sns.color_palette('deep',n_colors=5),
                  data=df)
# ax.set(ylim=(60, 75))

ax = sns.lineplot(x="dimension", y="accuracy", #hue="rank",
                  marker="X", linestyle='--', linewidth='5',
                  legend=False, #err_style=None,
                  palette=sns.color_palette('deep',n_colors=5),
                  data=df_d)

for i in range(1,2):
    ax.lines[i].set_linestyle('--')

# ax.set(ylim=(60.1, 65.5))
ax.grid(axis='x')





plt.figure()

ax2 = sns.lineplot(x="dimension", y="edit score", #hue="rank",
                  marker="o", 
                  legend=False, #err_style=None,
                  palette=sns.color_palette('deep',n_colors=5),
                  data=df)
# ax.set(ylim=(60, 75))

ax2 = sns.lineplot(x="dimension", y="edit score", #hue="rank",
                  marker="X", linestyle='--', linewidth='5',
                  legend=False, #err_style=None,
                  palette=sns.color_palette('deep',n_colors=5),
                  data=df_d)

for i in range(1,2):
    ax2.lines[i].set_linestyle('--')

# ax.set(ylim=(60.1, 65.5))
ax2.grid(axis='x')


plt.figure()

ax3 = sns.lineplot(x="dimension", y="f1 score", #hue="rank",
                  marker="o", 
                  legend=False, #err_style=None,
                  palette=sns.color_palette('deep',n_colors=5),
                  data=df)
# ax.set(ylim=(60, 75))

ax3 = sns.lineplot(x="dimension", y="f1 score", #hue="rank",
                  marker="X", linestyle='--', linewidth='5',
                  legend=False, #err_style=None,
                  palette=sns.color_palette('deep',n_colors=5),
                  data=df_d)

for i in range(1,2):
    ax3.lines[i].set_linestyle('--')

# ax.set(ylim=(60.1, 65.5))
ax3.grid(axis='x')



# ax = sns.relplot(x="dimension", y="edit score", hue="rank", #kind='line',
#                 markers=True, dashes=True,legend=False,err_style='bars',palette=sns.color_palette('deep',n_colors=5),
#                 data=df)
# # ax.set(ylim=(60, 66))
#
# ax = sns.relplot(x="dimension", y="f1 score", hue="rank", #kind='line',
#                 markers=True, dashes=True,legend='full',err_style='bars',palette=sns.color_palette('deep',n_colors=5),
#                 data=df)
# # ax.set(ylim=(65, 72))


plt.show()
plt.savefig('result.pdf', format='pdf')




