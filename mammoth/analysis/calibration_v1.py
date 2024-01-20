from __future__ import print_function
import os
import torch
import torch.nn.functional as F
from matplotlib.offsetbox import AnchoredText
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from datasets.seq_tinyimagenet import TinyImagenet


def eval_calibration(model, device, data_loader, axes, legend):

    n_bins = 20
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    lst_logits = []
    lst_labels = []

    with torch.no_grad():
        for input, label in data_loader:
            input, label = input.to(device), label.to(device)
            logits = model(input)
            lst_logits.append(logits.detach().cpu())
            lst_labels.append(label.detach().cpu())

    logits = torch.cat(lst_logits)
    labels = torch.cat(lst_labels)
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    lst_acc_in_bin = []
    lst_conf_in_bin = []
    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            lst_acc_in_bin.append(accuracy_in_bin.item())
            lst_conf_in_bin.append(avg_confidence_in_bin.item())
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            lst_acc_in_bin.append(0)
            lst_conf_in_bin.append(0)

    ece *= 100
    print('ECE: %s' % ece.item())
    col = (237, 129, 121)
    col = np.array(col) / 255
    col = tuple(col.tolist())
    x_axis = np.array(list(range(0, n_bins))) / n_bins
    axes.bar(x_axis, x_axis, align='edge', width=0.05, facecolor=(1, 0, 0, 0.3), edgecolor=col, hatch='//', label='Gap')
    axes.bar(x_axis, lst_acc_in_bin, align='edge', width=0.05, facecolor=(0, 0, 1, 0.5), edgecolor='blue', label='Outputs')
    x_axis = np.array(list(range(0, n_bins + 1))) / n_bins
    axes.plot(x_axis, x_axis, '--', color='k')
    axes.axis('equal')

    if legend:
        axes.legend(fontsize=10, loc='lower right')

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlabel('Confidence', fontsize=12,)
    anchored_text = AnchoredText('ECE=%.2f' % ece, loc='upper left', prop=dict(fontsize=10))
    axes.add_artist(anchored_text)

    return ece.item()


# =============================================================================
# Load Dataset
# =============================================================================
dataset_selection = 'cifar10'

if dataset_selection == 'cifar10':
    final_task = 5
    models_repo = r'/data/output/fahad.sarfraz/CLS-ER/cil_analysis/cifar10'
    lst_methods = {
        'der': 'der/task_models/seq-cifar10/der-c10-b-%s-r-0',
        'er': 'er/task_models/seq-cifar10/er-c10-b-%s-r-0',
        'cil-er': 'cil_er/task_models/seq-cifar10/er-c10-%s-param-v1-s-6',
    }

    # CIFAR 10
    TRANSFORM = transforms.Compose(
        [
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
         ]
    )

    dataset = CIFAR10('data', train=False, download=True, transform=TRANSFORM)

elif dataset_selection == 'tinyimg':

    final_task = 10
    models_repo = r'/data/output/fahad.sarfraz/cil_analysis/tinyimg'
    lst_methods = {
        'der': 'der/task_models/seq-tinyimg/der-tinyimg-b-%s-r-0',
        'er': 'er/task_models/seq-tinyimg/er-tinyimg-b-%s-r-0',
        'cil-er': 'cil_er/task_models/seq-tinyimg/er-tinyimg-%s-param-v2-s-0',
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
# lst_buffer_size = [200, 500, 5120]
lst_buffer_size = [500]
# lst_buffer_size = [5120]
device = 'cuda'

for buffer_size in lst_buffer_size:

    # buffer_size = 500
    print('=' * 50)
    print(f'Buffer Size = {buffer_size}')
    print('=' * 50)

    n_rows, n_cols = 1, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4), sharey=True)

    model_path = os.path.join(models_repo, lst_methods['er'] % buffer_size, f'task_{final_task}_model.ph')
    model = torch.load(model_path).to(device)
    model.eval()
    eval_calibration(model, 'cuda', data_loader, axes[0], False)
    axes[0].set_title('ER', fontdict={'fontsize': 12})

    model_path = os.path.join(models_repo, lst_methods['der'] % buffer_size, f'task_{final_task}_model.ph')
    model = torch.load(model_path).to(device)
    model.eval()
    eval_calibration(model, 'cuda', data_loader, axes[1], False)
    axes[1].set_title('DER++', fontdict={'fontsize': 12})

    model_path = os.path.join(models_repo, lst_methods['cil-er'] % buffer_size, f'task_{final_task}_stable_model.ph')
    model = torch.load(model_path).to(device)
    model.eval()
    eval_calibration(model, 'cuda', data_loader, axes[2], True)
    axes[2].set_title('CLS-ER', fontdict={'fontsize': 12})

    axes[0].set_ylabel('Accuracy', labelpad=11, fontsize=12, color='k')

    plt.show()
    fig.savefig(f'analysis/figures/calibration_plots/{dataset_selection}_{buffer_size}_v1.png', bbox_inches='tight')
    fig.savefig(f'analysis/figures/calibration_plots/{dataset_selection}_{buffer_size}_v1.pdf', bbox_inches='tight')
