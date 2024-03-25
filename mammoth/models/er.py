# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import numpy as np
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.clip_utils import get_similarity, get_similarity_new

LAYER_SIZE = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # GIL Param
    parser.add_argument('--gil_seed', type=int, default=1993, help='Seed value for GIL-CIFAR task sampling')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to use pretrain')
    parser.add_argument('--phase_class_upper', default=50, type=int, help='the maximum number of classes')
    parser.add_argument('--epoch_size', default=1000, type=int, help='Number of samples in one epoch')
    parser.add_argument('--pretrain_class_nb', default=0, type=int, help='the number of classes in first group')
    parser.add_argument('--weight_dist', default='unif', type=str, help='what type of weight distribution assigned to classes to sample (unif or longtail)')
    parser.add_argument('--tiny_imgnet_path', type=str, default='data')
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.current_task = 0
        self.layer_weights = [lr + ".1.conv2.weight" for lr in sorted(args.mask_layer)]
        layer_sizes = [par.size()[1] for name, par in self.net.named_parameters() if name in self.layer_weights]
        ones = torch.ones(3, 3)
        self.masks = [ones.repeat(layer_sizes[i], 1, 1) for i in range(len(layer_sizes))] 
        self.layers = sorted(args.mask_layer)
        self.n_neurons = [LAYER_SIZE[lr] for lr in self.layers]
        
    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        if self.current_task > 0:
            for name, param in self.net.named_parameters():
                if name in self.layer_weights: #change
                    ind = self.layer_weights.index(name)
                    mask = self.masks[ind].repeat(param.shape[0], 1,1,1)
                    param.grad *= mask

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()
    
    def end_task(self, dataset) -> None: #

        if self.current_task < (dataset.N_TASKS - 1):
            concept_set = get_concept(self.current_task)
            with open(concept_set, 'r') as f:
                words = (f.read()).split('\n')
            similarity_fn = "soft_wpmi"
            pool_mode = "avg"
            
            for i, layer in enumerate(self.layers):
            
                group_captions, groups = get_similarity_new(self.clip_model, self.net, [layer], concept_set,
                                                        self.args.batch_size, pool_mode, dataset, similarity_fn, device=self.device)
                
                print(group_captions)
                
                print(groups)
                
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

    # def end_task(self, dataset) -> None: #

    #     if self.current_task < (dataset.N_TASKS - 1):
    #         concept_set = get_concept(self.current_task)
    #         with open(concept_set, 'r') as f:
    #             words = (f.read()).split('\n')
    #         similarity_fn = "soft_wpmi"
    #         pool_mode = "avg"
            
    #         for i, layer in enumerate(self.layers):
            
    #             similarity, target_feats = get_similarity(self.clip_model, self.net, [layer], concept_set,
    #                                                     self.args.batch_size, pool_mode, dataset, similarity_fn, device=self.device)
                
    #             vals, ids = torch.max(similarity, dim=1)

    #             for class_id in range(dataset.N_CLASSES_PER_TASK):
    #                 task_ind = np.arange(len(vals))[ids.detach().cpu().numpy() == (dataset.i - dataset.N_CLASSES_PER_TASK + class_id)]
    #                 top5_sim, top5_ind = torch.topk(vals[task_ind], min(task_ind.shape[0], self.args.top_neurons))
    #                 neurons = task_ind[top5_ind.detach().cpu().numpy()]
    #                 print(neurons)
    #                 for neuron in neurons:
    #                     self.masks[i][neuron, :, :] *= self.args.grad_multiplier #torch.zeros(3, 3)
    #                     self.masks[i] = self.masks[i].to(self.device)

    #     model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
    #     os.makedirs(model_dir, exist_ok=True)
    #     torch.save(self.net, os.path.join(model_dir, f'task_{self.current_task}_model.ph'))

    #     self.current_task += 1
        
        
    # def end_task_alt(self, dataset) -> None: #

    #     if self.current_task < (dataset.N_TASKS - 1):
    #         concept_set = get_concept(self.current_task)
    #         with open(concept_set, 'r') as f:
    #             words = (f.read()).split('\n')
    #         similarity_fn = "soft_wpmi"
    #         pool_mode = "avg"
            
            
    #         for i, layer in enumerate(self.layers):
            
    #             similarity, target_feats = get_similarity(self.clip_model, self.net, [layer], concept_set,
    #                                                     self.args.batch_size, pool_mode, dataset, similarity_fn, device=self.device)
                
    #             vals, ids = torch.topk(similarity, 2, dim=1)
    #             diff_vals = torch.abs(vals[:, 0] - vals[:, 1])
                
    #             for class_id in range(dataset.N_CLASSES_PER_TASK):
    #                 task_ind = np.arange(len(vals))[ids.detach().cpu().numpy()[:, 0] == (dataset.i - dataset.N_CLASSES_PER_TASK + class_id)]
    #                 top5_ind = torch.topk(diff_vals[task_ind], min(task_ind.shape[0], self.args.top_neurons))[1].detach().cpu().numpy()
    #                 neurons = task_ind[top5_ind]
    #                 print(neurons)
    #                 for neuron in neurons:
    #                     self.masks[i][neuron, :, :] *= self.args.grad_multiplier #torch.zeros(3, 3)
    #                     self.masks[i] = self.masks[i].to(self.device)

    #     model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
    #     os.makedirs(model_dir, exist_ok=True)
    #     torch.save(self.net, os.path.join(model_dir, f'task_{self.current_task}_model.ph'))

    #     self.current_task += 1
        
    # def end_task2(self, dataset) -> None: #

    #     if self.current_task < (dataset.N_TASKS - 1):
    #         concept_set = get_concept(self.current_task)
    #         with open(concept_set, 'r') as f:
    #             words = (f.read()).split('\n')
    #         similarity_fn = "soft_wpmi"
    #         pool_mode = "avg"
            
    #         similarities = []
            
    #         for layer in self.layers:
            
    #             similarity, target_feats = get_similarity(self.clip_model, self.net, [layer], concept_set,
    #                                                     self.args.batch_size, pool_mode, dataset, similarity_fn, device=self.device)
                
    #             similarities.append(similarity)
                
    #         similarities = torch.cat(similarities)
    #         vals, ids = torch.topk(similarities, 2, dim=1)
    #         diff_vals = torch.abs(vals[:, 0] - vals[:, 1])
            
    #         for class_id in range(dataset.N_CLASSES_PER_TASK):
    #             task_ind = np.arange(len(vals))[ids.detach().cpu().numpy()[:, 0] == (dataset.i - dataset.N_CLASSES_PER_TASK + class_id)]
    #             top5_ind = torch.topk(diff_vals[task_ind], min(task_ind.shape[0], self.args.top_neurons))[1].detach().cpu().numpy()
    #             neurons = task_ind[top5_ind]
    #             previous = 0
    #             for i, sz in enumerate(self.n_neurons):
    #                 layer_neurons = (neurons < (sz + previous)) & (neurons >= previous)
    #                 top_neurons = neurons[layer_neurons] - previous
    #                 for neuron in top_neurons:
    #                     self.masks[i][neuron, :, :] *= self.args.grad_multiplier #torch.zeros(3, 3)
    #                     self.masks[i] = self.masks[i].to(self.device)

    #     model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
    #     os.makedirs(model_dir, exist_ok=True)
    #     torch.save(self.net, os.path.join(model_dir, f'task_{self.current_task}_model.ph'))

    #     self.current_task += 1



def get_concept(task):
    if task == 0:
        concept_set = 'data/cifar10_t0.txt'
    elif task == 1:
        concept_set = 'data/cifar10_t1.txt'
    elif task == 2:
        concept_set = 'data/cifar10_t2.txt'
    elif task == 3:
        concept_set = 'data/cifar10_t3.txt'
    elif task == 4:
        concept_set = 'data/cifar10_t4.txt'

    return concept_set