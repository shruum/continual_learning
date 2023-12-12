import matplotlib.pyplot as plt
import numpy as np
import os

results_dir = r'/data/output/fahad.sarfraz/cil_analysis/tinyimg/cil_er/results/class-il/seq-tinyimg/dualmeanerv6'
exp_id = 'er-tinyimg-5120-param-v2-s-0'
num_tasks = 10

# results_dir = r'/data/output/fahad.sarfraz/cil_analysis/cifar10/cil_er/results/class-il/seq-cifar10/dualmeanerv6'
# exp_id = 'er-c10-200-param-v1-s-0'
# num_tasks = 5

perf_path = os.path.join(results_dir, exp_id, 'task_performance.txt')
np_perf = np.loadtxt(perf_path)

n_rows, n_cols = 1, 3
fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 5), sharey=True)

x_labels = [f"Task {i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After Task {i}" for i in range(1, num_tasks + 1)]

perf_path = os.path.join(results_dir, exp_id + '_stable_ema_model', 'task_performance.txt')
np_perf = np.loadtxt(perf_path)
im = ax[0].imshow(np_perf)
ax[0].set_xticks(np.arange(len(x_labels)))
ax[0].set_yticks(np.arange(len(y_labels)))
ax[0].set_xticklabels(x_labels)
ax[0].set_yticklabels(y_labels)
plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in range(len(x_labels)):
    for j in range(len(y_labels)):
        text = ax[0].text(j, i, np_perf[i, j], ha="center", va="center", color="w")
ax[0].set_title('Stable Model', fontsize=17)


perf_path = os.path.join(results_dir, exp_id , 'task_performance.txt')
np_perf = np.loadtxt(perf_path)
im = ax[1].imshow(np_perf)
ax[1].set_xticks(np.arange(len(x_labels)))
ax[1].set_yticks(np.arange(len(y_labels)))
ax[1].set_xticklabels(x_labels)
ax[1].set_yticklabels(y_labels)
plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in range(len(x_labels)):
    for j in range(len(y_labels)):
        text = ax[1].text(j, i, np_perf[i, j], ha="center", va="center", color="w")
ax[1].set_title('Working Model', fontsize=17)

perf_path = os.path.join(results_dir, exp_id + '_plastic_ema_model', 'task_performance.txt')
np_perf = np.loadtxt(perf_path)
im = ax[2].imshow(np_perf)
ax[2].set_xticks(np.arange(len(x_labels)))
ax[2].set_yticks(np.arange(len(y_labels)))
ax[2].set_xticklabels(x_labels)
ax[2].set_yticklabels(y_labels)
plt.setp(ax[2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in range(len(x_labels)):
    for j in range(len(y_labels)):
        text = ax[2].text(j, i, np_perf[i, j], ha="center", va="center", color="w")
ax[2].set_title('Plastic Model', fontsize=17)

fig.tight_layout()
plt.show()

fig.savefig(f'analysis/figures/task_probabilities/{exp_id}.png', bbox_inches='tight')
fig.savefig(f'analysis/figures/task_probabilities/{exp_id}.pdf', bbox_inches='tight')
