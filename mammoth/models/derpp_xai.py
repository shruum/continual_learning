# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
import os
import numpy as np
from utils.clip_utils import get_similarity, get_neurons

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_xai_args(parser)
    #
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    # GIL Param
    parser.add_argument('--gil_seed', type=int, default=1993, help='Seed value for GIL-CIFAR task sampling')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to use pretrain')
    parser.add_argument('--phase_class_upper', default=50, type=int, help='the maximum number of classes')
    parser.add_argument('--epoch_size', default=1000, type=int, help='Number of samples in one epoch')
    parser.add_argument('--pretrain_class_nb', default=0, type=int, help='the number of classes in first group')
    parser.add_argument('--weight_dist', default='unif', type=str, help='what type of weight distribution assigned to classes to sample (unif or longtail)')
    parser.add_argument('--tiny_imgnet_path', type=str, default='data')
    return parser


class DerppXAI(ContinualModel):
    NAME = 'derpp_xai'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DerppXAI, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0

        ones = torch.ones(3, 3)
        print(args.mask_layer)
        self.layer_weights = [lr + ".1.conv2.weight" for lr in sorted(args.mask_layer)]
        layer_sizes = [par.size()[1] for name, par in self.net.named_parameters() if name in self.layer_weights]
        self.masks = [ones.repeat(layer_sizes[i], 1, 1) for i in range(len(layer_sizes))]
        self.layers = sorted(args.mask_layer)
        self.neurons = get_neurons(self.layers)

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()

        if self.current_task > 0:
            for name, param in self.net.named_parameters():
                if name in self.layer_weights:  # change
                    ind = self.layer_weights.index(name)
                    mask = self.masks[ind].repeat(self.neurons[name[0:6]], 1, 1, 1)
                    param.grad *= mask

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item()

    def end_task(self, dataset) -> None:
        # self.current_task += 1
        model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.net, os.path.join(model_dir, f'task_{self.current_task}_model.ph'))

        if self.current_task < (dataset.N_TASKS - 1):
            concept_set = dataset.get_concept(self.current_task)
            with open(concept_set, 'r') as f:
                words = (f.read()).split('\n')
            similarity_fn = "soft_wpmi"
            pool_mode = "avg"

            self.clip_model.eval()
            self.net.eval()

            for i, layer in enumerate(self.layers):
                similarity, target_feats = get_similarity(self.clip_model, self.net, [layer], concept_set,
                                                        self.args.batch_size, pool_mode, dataset, similarity_fn, device=self.device)
                print("SIIIM, ", similarity.shape)
                vals, ids = torch.max(similarity, dim=1)

                for class_id in range(dataset.N_CLASSES_PER_TASK):
                    task_ind = np.arange(len(vals))[ids.detach().cpu().numpy() == (dataset.i - dataset.N_CLASSES_PER_TASK + class_id)]
                    top5_sim, top5_ind = torch.topk(vals[task_ind], min(task_ind.shape[0], self.args.top_neurons))
                    neurons = task_ind[top5_ind.detach().cpu().numpy()]
                    print(neurons)
                    for neuron in neurons:
                        self.masks[i][neuron, :, :] *= self.args.grad_multiplier #torch.zeros(3, 3)
                        self.masks[i] = self.masks[i].to(self.device)

        model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.net, os.path.join(model_dir, f'task_{self.current_task}_model.ph'))

        self.current_task += 1
        self.net.train()
