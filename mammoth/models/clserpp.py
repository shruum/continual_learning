import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from models.utils.distillation_losses import at_loss
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
from torch.distributions.beta import Beta
import numpy as np


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # Consistency Regularization Weight
    parser.add_argument('--reg_weight', type=float, default=0.1)
    parser.add_argument('--at_weight', type=float, default=0.1)
    parser.add_argument('--prot_weight', type=float, default=0.1)
    parser.add_argument('--prot_level', type=str, default='buffer', choices=['buffer', 'batch'])

    # Stable Model parameters
    parser.add_argument('--stable_model_update_freq', type=float, default=0.70)
    parser.add_argument('--stable_model_alpha', type=float, default=0.999)

    # Plastic Model Parameters
    parser.add_argument('--plastic_model_update_freq', type=float, default=0.90)
    parser.add_argument('--plastic_model_alpha', type=float, default=0.999)

    # Feature Distillation
    parser.add_argument('--feat_level', type=int, default=4)
    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class CLSERPP(ContinualModel):
    NAME = 'clser'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(CLSERPP, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        # Initialize plastic and stable model
        self.plastic_model = deepcopy(self.net).to(self.device)
        self.stable_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = args.reg_weight
        # set parameters for plastic model
        self.plastic_model_update_freq = args.plastic_model_update_freq
        self.plastic_model_alpha = args.plastic_model_alpha
        # set parameters for stable model
        self.stable_model_update_freq = args.stable_model_update_freq
        self.stable_model_alpha = args.stable_model_alpha
        # feature level
        self.feat_level = args.feat_level
        self.at_weight = args.at_weight
        self.prot_weight = args.prot_weight
        self.prot_level = args.prot_level

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0

        self.prot_loss = nn.MSELoss()


    @staticmethod
    def get_classwise_acc(output, target, num_classes):
        confusion_matrix = torch.zeros(num_classes, num_classes).to(output.device)
        _, preds = torch.max(output, 1)

        for t, p in zip(target.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        classwise_acc = confusion_matrix.diag() / confusion_matrix.sum(1)

        return classwise_acc

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        loss = 0

        if not self.buffer.is_empty():

            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            stable_model_feats, stable_model_logits = self.stable_model.extract_features(buf_inputs)
            plastic_model_feats, plastic_model_logits = self.plastic_model.extract_features(buf_inputs)

            stable_model_prob = F.softmax(stable_model_logits, 1)
            plastic_model_prob = F.softmax(plastic_model_logits, 1)

            label_mask = F.one_hot(buf_labels, num_classes=stable_model_logits.shape[-1]) > 0
            sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
            sel_idx = sel_idx.unsqueeze(1)

            model_feats, model_logits = self.net.extract_features(buf_inputs)

            stable_model_feats, plastic_model_feats, model_feats = \
                stable_model_feats[self.feat_level], plastic_model_feats[self.feat_level], model_feats[self.feat_level]

            # Consistency Loss on the logits
            ema_logits = torch.where(
                sel_idx,
                stable_model_logits,
                plastic_model_logits,
            )

            l_cons = torch.mean(self.consistency_loss(model_logits, ema_logits.detach()))

            # Attention loss on the features
            ema_feats = torch.where(
                sel_idx.unsqueeze(-1).unsqueeze(-1),
                stable_model_feats,
                plastic_model_feats,
            )

            l_at = at_loss(model_feats, ema_feats)

            # Class Prototypes
            if self.prot_level == 'buffer':
                # print('Applying Prototype loss at Buffer Level')
                class_prot_shape = [self.num_classes] + list(model_feats.shape)[1:]
                counts = torch.ones(self.num_classes) * 1e-5

                model_class_prot = torch.zeros(class_prot_shape).to(self.device)
                stable_model_class_prot = torch.zeros(class_prot_shape).to(self.device)
                plastic_model_class_prot = torch.zeros(class_prot_shape).to(self.device)

                for sample_idx in range(model_feats.shape[0]):
                    sample_label = int(labels[sample_idx])
                    counts[sample_label] += 1
                    model_class_prot[sample_label] += model_feats[sample_idx]
                    stable_model_class_prot[sample_label] += stable_model_feats[sample_idx]
                    plastic_model_class_prot[sample_label] += plastic_model_feats[sample_idx]

                counts = counts.to(self.device)
                counts = counts.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                model_class_prot /= counts
                stable_model_class_prot /= counts
                plastic_model_class_prot /= counts

                stable_classwise = self.get_classwise_acc(stable_model_logits, labels, self.num_classes)
                plastic_classwise = self.get_classwise_acc(plastic_model_logits, labels, self.num_classes)

                feat_sel_idx = stable_classwise > plastic_classwise

                ema_class_prototypes = torch.where(
                    feat_sel_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                    stable_model_class_prot,
                    plastic_model_class_prot,
                )

                l_prot = self.prot_loss(ema_class_prototypes.detach(), model_class_prot)

                l_reg = self.args.reg_weight * l_cons + self.at_weight * l_at + self.prot_weight * l_prot

                if hasattr(self, 'writer'):
                    self.writer.add_scalar(f'Task {self.current_task}/l_prot', l_prot.item(), self.iteration)

            else:
                l_reg = self.args.reg_weight * l_cons + self.at_weight * l_at

            loss += l_reg

            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

            # Log values
            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_cons', l_cons.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/l_at', l_at.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/l_reg', l_reg.item(), self.iteration)


        model_feats, model_logits = self.net.extract_features(inputs)

        ce_loss = self.loss(model_logits, labels)
        loss += ce_loss

        # Class Prototypes
        if self.prot_level == 'batch':
            # print('Applying Prototype loss at Batch Level')
            stable_model_feats, stable_model_logits = self.stable_model.extract_features(inputs)
            plastic_model_feats, plastic_model_logits = self.plastic_model.extract_features(inputs)

            stable_model_feats, plastic_model_feats, model_feats = \
                stable_model_feats[self.feat_level], plastic_model_feats[self.feat_level], model_feats[self.feat_level]


            class_prot_shape = [self.num_classes] + list(model_feats.shape)[1:]
            counts = torch.ones(self.num_classes) * 1e-5

            model_class_prot = torch.zeros(class_prot_shape).to(self.device)
            stable_model_class_prot = torch.zeros(class_prot_shape).to(self.device)
            plastic_model_class_prot = torch.zeros(class_prot_shape).to(self.device)

            for sample_idx in range(model_feats.shape[0]):
                sample_label = int(labels[sample_idx])
                counts[sample_label] += 1
                model_class_prot[sample_label] += model_feats[sample_idx]
                stable_model_class_prot[sample_label] += stable_model_feats[sample_idx]
                plastic_model_class_prot[sample_label] += plastic_model_feats[sample_idx]

            counts = counts.to(self.device)
            counts = counts.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            model_class_prot /= counts
            stable_model_class_prot /= counts
            plastic_model_class_prot /= counts

            stable_classwise = self.get_classwise_acc(stable_model_logits, labels, self.num_classes)
            plastic_classwise = self.get_classwise_acc(plastic_model_logits, labels, self.num_classes)

            feat_sel_idx = stable_classwise > plastic_classwise

            ema_class_prototypes = torch.where(
                feat_sel_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                stable_model_class_prot,
                plastic_model_class_prot,
            )

            l_prot = self.prot_loss(ema_class_prototypes.detach(), model_class_prot)

            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_prot', l_prot.item(), self.iteration)

            loss += (self.prot_weight * l_prot)



        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size],
        )

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.plastic_model_update_freq:
            self.update_plastic_model_variables()

        if torch.rand(1) < self.stable_model_update_freq:
            self.update_stable_model_variables()

        return loss.item()

    def update_plastic_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.plastic_model_alpha)
        for ema_param, param in zip(self.plastic_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_stable_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1),  self.stable_model_alpha)
        for ema_param, param in zip(self.stable_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
