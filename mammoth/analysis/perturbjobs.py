# THIS IS FOR LAUNCHING YOUR EXPERIMENTS
# IT'S JUST A GLORIFIED SHELL SCRIPT
import os
from tqdm import tqdm

maxcheckpoints = ['der_seq-cifar10_0500_04.pt', 'derpp_seq-cifar10_0500_04.pt', 'fdr_seq-cifar10_0500_04.pt', 'er_seq-cifar10_0500_04.pt']
maxcheckpoints = maxcheckpoints * 3 # we did 3 experiment for each checkpoint

current_dataset = None
for checkpoint in tqdm(maxcheckpoints):
    current_dataset = checkpoint.split('_')[1]

    try:
        command = 'python3 perturbation.py --load_best_args --checkpoint=<PATH TO OUR CHECKPOINTS>/%s --redund=1' % checkpoint

        print()
        os.system(command)
    except:
        exit()
