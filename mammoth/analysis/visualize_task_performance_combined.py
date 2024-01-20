import matplotlib.pyplot as plt
import numpy as np
import os



dataset = 'tinyimg'


if dataset == 'tinyimg':
    # results_dir = r'/data/output/fahad.sarfraz/cil_analysis/tinyimg/cil_er/results/class-il/seq-tinyimg/dualmeanerv6'
    # exp_id = 'er-tinyimg-%s-param-v3-s-0'

    results_dir = r'/data/output/fahad.sarfraz/cls_er_imagenet_final/results/class-il/seq-tinyimg/clser'
    exp_id = 'er-tinyimg-%s-param-v3-s-4'

    num_tasks = 10

else:
    results_dir = r'/data/output/fahad.sarfraz/cil_analysis/cifar10/cil_er/results/class-il/seq-cifar10/dualmeanerv6'
    exp_id = 'er-c10-%s-param-v1-s-0'
    num_tasks = 5

x_labels = [f"Task {i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After Task {i}" for i in range(1, num_tasks + 1)]

n_rows, n_cols = 3, 3
fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15), sharey=True, sharex=True)

lst_buffer_size = [200, 500, 5120]

for n, buffer_size in enumerate(lst_buffer_size):
    perf_path = os.path.join(results_dir, exp_id % buffer_size + '_stable_ema_model', 'task_performance.txt')
    np_perf = np.loadtxt(perf_path)
    im = ax[n, 0].imshow(np_perf)
    ax[n, 0].set_xticks(np.arange(len(x_labels)))
    ax[n, 0].set_yticks(np.arange(len(y_labels)))
    ax[n, 0].set_xticklabels(x_labels)
    ax[n, 0].set_yticklabels(y_labels)
    plt.setp(ax[n, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax[n, 0].text(j, i, np_perf[i, j], ha="center", va="center", color="w")


    perf_path = os.path.join(results_dir, exp_id % buffer_size, 'task_performance.txt')
    np_perf = np.loadtxt(perf_path)
    im = ax[n, 1].imshow(np_perf)
    ax[n, 1].set_xticks(np.arange(len(x_labels)))
    ax[n, 1].set_yticks(np.arange(len(y_labels)))
    ax[n, 1].set_xticklabels(x_labels)
    ax[n, 1].set_yticklabels(y_labels)
    plt.setp(ax[n, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax[n, 1].text(j, i, np_perf[i, j], ha="center", va="center", color="w")


    perf_path = os.path.join(results_dir, exp_id % buffer_size+ '_plastic_ema_model', 'task_performance.txt')
    np_perf = np.loadtxt(perf_path)
    im = ax[n, 2].imshow(np_perf)
    ax[n, 2].set_xticks(np.arange(len(x_labels)))
    ax[n, 2].set_yticks(np.arange(len(y_labels)))
    ax[n, 2].set_xticklabels(x_labels)
    ax[n, 2].set_yticklabels(y_labels)
    plt.setp(ax[n, 2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax[n, 2].text(j, i, np_perf[i, j], ha="center", va="center", color="w")




ax[0, 0].set_title('Stable Model', fontsize=17)
ax[0, 1].set_title('Working Model', fontsize=17)
ax[0, 2].set_title('Plastic Model', fontsize=17)
ax[0, 0].set_ylabel('Buffer 200', labelpad=11, fontsize=17, color='k')
ax[1, 0].set_ylabel('Buffer 500', labelpad=11, fontsize=17, color='k')
ax[2, 0].set_ylabel('Buffer 5120', labelpad=11, fontsize=17, color='k')
fig.tight_layout()
plt.show()


fig.savefig(f'analysis/figures/task_performance/{dataset}.png', bbox_inches='tight')
fig.savefig(f'analysis/figures/task_performance/{dataset}.pdf', bbox_inches='tight')
