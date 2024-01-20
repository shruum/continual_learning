from __future__ import print_function
import os
import torch
import torch.nn.functional as F
from matplotlib.offsetbox import AnchoredText
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from datasets import ContinualDataset
from datasets import get_dataset
from argparse import Namespace
from datasets.seq_tinyimagenet import TinyImagenet
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def get_task_probabilities(model, device, data_loader, task_dist):
    model.eval()
    lst_logits = []

    with torch.no_grad():
        for input, label in data_loader:
            input, label = input.to(device), label.to(device)
            logits = model(input)

            lst_logits.append(logits.detach().cpu())

    logits = torch.cat(lst_logits).to(device)
    softmax_scores = F.softmax(logits, dim=1)

    lst_prob = []
    for task in task_dist:
        prob = torch.mean(softmax_scores[:, task_dist[task][0]: task_dist[task][1]])
        lst_prob.append(prob.item())

    np_prob = np.array(lst_prob)
    np_prob = np_prob / np.sum(np_prob)

    return np_prob


# =============================================================================
# Load Dataset
# =============================================================================
models_repo = r'/data/output/fahad.sarfraz/cil_analysis/tinyimg'
lst_methods = {
    'der': 'der/task_models/seq-tinyimg/der-tinyimg-b-%s-r-0',
    'er': 'er/task_models/seq-tinyimg/er-tinyimg-b-%s-r-0',
    'cls-er': 'cil_er/task_models/seq-tinyimg/er-tinyimg-%s-param-v2-s-0',
}

TRANSFORM = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4802, 0.4480, 0.3975),
                          (0.2770, 0.2691, 0.2821))])

dataset = TinyImagenet(r'/data/output/fahad.sarfraz/datasets/TINYIMG', train=False, download=True, transform=TRANSFORM)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# =============================================================================
# Evaluate Calibration Plots
# =============================================================================
lst_buffer_size = [200, 500, 5120]
# lst_buffer_size = [200]
# lst_buffer_size = [5120]
device = 'cuda'

task_dist = {
    'task1': (0, 20),
    'task2': (20, 40),
    'task3': (40, 60),
    'task4': (60, 80),
    'task5': (80, 100),
    'task6': (100, 120),
    'task7': (120, 140),
    'task8': (140, 160),
    'task9': (160, 180),
    'task10': (180, 200),
}

lst_colors = [
    '#fafa6e',
    '#c4ec74',
    '#92dc7e',
    '#64c987',
    '#39b48e',
    '#089f8f',
    '#00898a',
    '#08737f',
    '#215d6e',
    '#2a4858',
]

for buffer_size in lst_buffer_size:
    # buffer_size = 500
    print('=' * 50)
    print(f'Buffer Size = {buffer_size}')
    print('=' * 50)


    model_path = os.path.join(models_repo, lst_methods['er'] % buffer_size, f'task_10_model.ph')
    model = torch.load(model_path).to(device)
    er_prob = get_task_probabilities(model, 'cuda', data_loader, task_dist)


    model_path = os.path.join(models_repo, lst_methods['der'] % buffer_size, f'task_10_model.ph')
    model = torch.load(model_path).to(device)
    der_prob = get_task_probabilities(model, 'cuda', data_loader, task_dist)

    model_path = os.path.join(models_repo, lst_methods['cls-er'] % buffer_size, f'task_10_stable_model.ph')
    model = torch.load(model_path).to(device)
    cls_prob = get_task_probabilities(model, 'cuda', data_loader, task_dist)

    prob = np.vstack((er_prob, der_prob, cls_prob))
    n_methods, n_tasks = prob.shape

    fig, ax = plt.subplots(figsize=(8, 5))
    barWidth = 0.07
    for i in range(n_tasks):
        x = np.arange(n_methods) + i * barWidth
        plt.bar(x, prob[:, i], color=lst_colors[i], width=barWidth, label=f'Task {i + 1}')

    plt.ylabel('Task Probability', fontsize=17)
    plt.xticks([r + 5 * barWidth for r in range(n_methods)], ['ER', 'DER++', 'CLS-ER'], fontsize=15)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.show()

    fig.savefig(f'analysis/figures/task_probability/tinyimg_{buffer_size}.png', bbox_inches='tight')
    fig.savefig(f'analysis/figures/task_probability/tinyimg_{buffer_size}.pdf', bbox_inches='tight')
