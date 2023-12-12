import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# lst_colors = [
#     '#f9dbbd',
#     '#ffa5ab',
#     '#da627d',
#     '#a53860',
#     '#450920',
# ]
lst_colors = [
    # '#f9dbbd',
    "#ffe5d9",
    "#ffcfd2",
    '#ffa5ab',
    '#da627d',
    '#a53860',
    # '#450920',
]
from matplotlib.colors import LinearSegmentedColormap
custom1 = LinearSegmentedColormap.from_list(
    name='pink',
    colors=lst_colors,
)

dataset = 'cifar10'

if dataset == 'cifar10':
    lst_methods = {
        'sgd': '/volumes1/xai_cl/resnet_l64/sgd/sgd_results',
        'er': '/volumes1/xai_cl/resnet_l64/er/er_results/base_cif10_er_resl64',
        'der': '/volumes1/xai_cl/resnet_l64/der/der_results/base_cif10_derp_resl64',
        'cls-er': '/volumes1/xai_cl/resnet_l64/clser/clser_res/',

    }
    num_tasks = 5
    annot = True


x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After T{i}" for i in range(1, num_tasks + 1)]

n_rows, n_cols = 1, 4
fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 13), sharey=True, sharex=True)

annot = True
fmt = '.1f'
# fmt = '%d'
font = 22

lst_method = ['sgd', 'er', 'der', 'cls-er']
buffer_size = 200

# Get Max and Min
v_max = 0
v_min = 1000
for n, method in enumerate(lst_method):
    perf_path = os.path.join(lst_methods[method] , 'task_performance.txt')

    np_perf = np.loadtxt(perf_path)
    matrix = np.triu(np.ones_like(np_perf)) - np.identity(np_perf.shape[0])
    max, min = np_perf.max(), np_perf.min()

    if v_max < max:
        v_max = max
    if v_min > min:
        v_min = min


x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After T{i}" for i in range(1, num_tasks + 1)]

k = 0
for n, method in enumerate(lst_method):
    perf_path = os.path.join(lst_methods[method] , 'task_performance.txt')

    np_perf = np.loadtxt(perf_path)

    # if n > 2:
    #     k = 1
    #     n -= 3
    # if n == 2:
    #     #cbar_ax = fig.add_axes([0.91, 0.18, 0.02, 0.6])
    #     im = sns.heatmap(np_perf, ax=ax[n], vmax=v_max, vmin=v_min, mask=matrix, annot=annot, cmap=custom1, alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": font}) #, cbar_ax=cbar_ax)
    # else:
    im = sns.heatmap(np_perf, ax=ax[n], vmax=v_max, vmin=v_min, mask=matrix, annot=annot, cmap=custom1, alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": font})
    ax[n].set_xticks(np.arange(len(x_labels)) + 0.5)
    ax[n].set_yticks(np.arange(len(y_labels)) + 0.5)
    ax[n].set_xticklabels(x_labels, ha='center', fontsize=font)
    ax[n].set_yticklabels(y_labels, rotation=0, va='center', fontsize=font)
    ax[n].set_aspect('equal', adjustable='box')

    ax[n].axhline(y=0, color='k', linewidth=1)
    ax[n].axhline(y=np_perf.shape[1], color='k', linewidth=2)
    ax[n].axvline(x=0, color='k', linewidth=1)
    ax[n].axvline(x=np_perf.shape[1], color='k', linewidth=2)



ax[0].set_title('SGD', fontsize=font+3)
ax[1].set_title('ER++', fontsize=font+3)
ax[2].set_title('DER++', fontsize=font+3)
ax[3].set_title('CLS-ER', fontsize=font+3)
# ax[1][0].set_position([0.24,0.125,0.228,0.343])
# ax[1][1].set_position([0.55,0.125,0.228,0.343])


# fig.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.25)
plt.show()

fig.savefig(f'/volumes1/xai_cl/task_all_cif10.pdf', bbox_inches='tight', dpi=500)
# fig.savefig(f'/volumes2/continual_learning/paper/analysis/new/task_all_dom_500.pdf', dpi=600, bbox_inches='tight')