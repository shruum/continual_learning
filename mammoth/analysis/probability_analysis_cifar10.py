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
from torchvision.datasets import CIFAR10
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


def get_task_probabilities_ensemble(model1, model2, device, data_loader, task_dist):

    model1.eval()
    model2.eval()

    lst_logits = []

    with torch.no_grad():
        for input, label in data_loader:
            input, label = input.to(device), label.to(device)
            logits1 = model1(input)
            logits2 = model2(input)

            logits = logits1 + logits2

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
# CIFAR 10
TRANSFORM = transforms.Compose(
    [
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2615))
     ]
)

dataset = CIFAR10('data', train=False, download=True, transform=TRANSFORM)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# =============================================================================
# Evaluate Calibration Plots
# =============================================================================
models_repo = r'/data/output/fahad.sarfraz/cil_analysis/cifar10'
lst_buffer_size = [200, 500, 5120]
# lst_buffer_size = [200]
# lst_buffer_size = [5120]
device = 'cuda'

lst_methods = {
    'der': 'der/task_models/seq-cifar10/der-c10-b-%s-r-0',
    'er': 'er/task_models/seq-cifar10/er-c10-b-%s-r-0',
    'cls-er': 'cil_er/task_models/seq-cifar10/er-c10-%s-param-v1-s-0',
}

lst_tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5']

task_dist = {
    'task1': (0, 2),
    'task2': (2, 4),
    'task3': (4, 6),
    'task4': (6, 8),
    'task5': (8, 10),
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

    fig, ax = plt.subplots(figsize=(15, 10), sharey=True)

    ind = np.arange(len(lst_tasks))
    width = 0.1

    results = {}
    model_path = os.path.join(models_repo, lst_methods['er'] % buffer_size, f'task_5_model.ph')
    model = torch.load(model_path).to(device)
    er_prob = get_task_probabilities(model, 'cuda', data_loader, task_dist)
    # ax.bar(ind, task_prob, width, label='ER', color='firebrick')

    model_path = os.path.join(models_repo, lst_methods['der'] % buffer_size, f'task_5_model.ph')
    model = torch.load(model_path).to(device)
    der_prob= get_task_probabilities(model, 'cuda', data_loader, task_dist)
    # ax.bar(ind + width, task_prob, width, label='DER++', color='steelblue')

    model_path = os.path.join(models_repo, lst_methods['cls-er'] % buffer_size, f'task_5_stable_model.ph')
    model = torch.load(model_path).to(device)
    cls_prob = get_task_probabilities(model, 'cuda', data_loader, task_dist)
    # ax.bar(ind + 2 * width, task_prob, width, label='CLS-ER-Stable', color='turquoise')

    # plastic_model_path = os.path.join(models_repo, lst_methods['cls-er'] % buffer_size, f'task_5_plastic_model.ph')
    # plastic_model = torch.load(plastic_model_path).to(device)
    # results['CLS-ER-Plastic'] = get_task_probabilities(plastic_model, 'cuda', data_loader, task_dist)
    # ax.bar(ind + 3 * width, task_prob, width, label='CLS-ER-Plastic', color='rebeccapurple')

    # model_path = os.path.join(models_repo, lst_methods['cls-er'] % buffer_size, f'task_5_model.ph')
    # model = torch.load(model_path).to(device)
    # results['CLS-ER-Model'] = get_task_probabilities(model, 'cuda', data_loader, task_dist)
    # ax.bar(ind + 4 *width, task_prob, width, label='CLS-ER-Model', color='mediumpurple')

    # task_prob = get_task_probabilities_ensemble(plastic_model, stable_model, 'cuda', data_loader, task_dist)
    # ax.bar(ind + 5 * width, task_prob, width, label='CLS-ER-Ensemble', color='green')
    prob = np.vstack((er_prob, der_prob, cls_prob))
    n_methods, n_tasks = prob.shape


    fig, ax = plt.subplots(figsize=(7, 5))
    barWidth = 0.12
    for i in range(n_tasks):
        x = np.arange(n_methods) + i * barWidth
        plt.bar(x, prob[:, i], color=lst_colors[i], width=barWidth, label=f'Task {i + 1}')

    plt.ylabel('Task Probability', fontsize=17)
    plt.xticks([r + 2 * barWidth for r in range(n_methods)], ['ER', 'DER++', 'CLS-ER',], fontsize=15)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.show()
    fig.savefig(f'analysis/figures/task_probability/c10_{buffer_size}.png', bbox_inches='tight')
    fig.savefig(f'analysis/figures/task_probability/c10_{buffer_size}.pdf', bbox_inches='tight')



