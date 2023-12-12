# THIS MUST BE PLACED IN MAMMOTH
# ROOT FOLDER (ensure that PYTHONPATH includes the root directory)

import importlib
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed, base_path
from backbone.ResNet import GlobalNorm
from backbone.utils.modules import BatchRenormalization2D
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import datetime

parser = ArgumentParser(description='mammoth', allow_abbrev=False)
# parser.add_argument('--model', type=str, required=False,
#                     help='Model name.', choices=get_all_models())
parser.add_argument('--load_best_args', action='store_true',
                    help='Loads the best arguments for each method, '
                         'dataset and memory buffer.')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--redund', type=int, required=True)
torch.set_num_threads(2)
add_management_args(parser)
args = parser.parse_known_args()[0]
checkpoint = args.checkpoint
redund = args.redund
if args.renorm:
    GlobalNorm(BatchRenormalization2D)
else:
    GlobalNorm(nn.BatchNorm2d)
mod = importlib.import_module('models.' + checkpoint.split('/')[-1].split('_')[0])

if args.load_best_args:
    parser.add_argument('--model', type=str, default=checkpoint.split('/')[-1].split('_')[0],
                    help='Model name.', choices=get_all_models())
    parser.add_argument('--dataset', type=str, default=checkpoint.split('/')[-1].split('_')[1],
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    if hasattr(mod, 'Buffer'):
        parser.add_argument('--buffer_size', type=int, default=int(checkpoint.split('/')[-1].split('_')[2]),
                            help='The size of the memory buffer.')
    args = parser.parse_args()
    best = best_args[args.dataset][args.model.split('_')[0]]
    if hasattr(args, 'buffer_size'):
        best = best[args.buffer_size]
    else:
        best = best[-1]
    for key, value in best.items():
        setattr(args, key, value)
else:
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    args = parser.parse_args()

if args.seed is not None:
    set_random_seed(args.seed)

if args.model == 'mer': setattr(args, 'batch_size', 1)
dataset = get_dataset(args)
lossf = dataset.get_loss()
backbone = dataset.get_backbone()
loss = dataset.get_loss()
model = get_model(args, backbone, loss, dataset.get_transform())
model.net.to(model.device)
model.net.eval()
training_loaders = []
test_loaders = []
for i in range(dataset.N_TASKS):
    trl, tel = dataset.get_data_loaders()
    training_loaders.append(trl)
    test_loaders.append(tel)

def perturbate(sigma):
    model.net.load_state_dict(torch.load(checkpoint, map_location=torch.device(model.device)))
    noise = np.random.normal(loc=0, scale=sigma, size=model.net.get_params().shape).astype(np.float32)
    model.net.set_params(model.net.get_params() + torch.tensor(noise).to(model.device))
    taskresults, lossresults = [], []
    for loader in training_loaders:
        correct = 0
        total = 0
        loss = 0
        for i in loader:
            x, y, _ = i
            
            outputs = model(x.to(model.device))
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == y.to(model.device)).item()
            total += y.shape[0]
            loss += lossf(outputs, y.to(model.device)).item()
        taskresults.append(correct / total)
        lossresults.append(loss)
    return taskresults, lossresults


noises = {
    'seq-cifar10' : np.arange(0.000, 0.0402, step=0.001),
    'seq-tinyimg' : np.arange(0.000, 0.0202, step=0.001),
    'seq-mnist' : np.arange(0.000, 0.101, step=0.005),
    'perm-mnist' : np.arange(0.000, 0.101, step=0.005),
    'rot-mnist' : np.arange(0.000, 0.101, step=0.005),
}

resultsAcc = []
resultsLoss = []
for redund in tqdm(range(redund), leave=False):
    tempres = []
    temploss = []
    
    for asigma in tqdm(noises[args.dataset], leave=False):
        tr, lr = perturbate(asigma)
        tempres.append(tr)
        temploss.append(lr)
    resultsAcc.append(tempres)
    resultsLoss.append(temploss)

resultsAcc = np.array(resultsAcc)
resultsLoss = np.array(resultsLoss)
# results are saved here
np.save(base_path() + 'perturbation/acc_%s_%s_%04d_%s.npy' % (model.NAME, dataset.NAME, args.buffer_size, str(datetime.datetime.now()).replace(':', '.')), resultsAcc)
np.save(base_path() + 'perturbation/loss_%s_%s_%04d_%s.npy' % (model.NAME, dataset.NAME, args.buffer_size, str(datetime.datetime.now()).replace(':', '.')), resultsLoss)
