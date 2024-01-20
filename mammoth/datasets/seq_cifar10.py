# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from datasets.seq_tinyimagenet import base_path
from PIL import Image
import numpy as np
from datasets.utils.validation import get_train_val, get_train_data_clip
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders, store_clip_masked_loaders, store_xai_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize


class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR10(ContinualDataset):

    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
             ]
    )
    TRANSFORM_CLIP = transforms.Compose(
            [transforms.Resize(224),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
             ]
    )
    TRANSFORM_XAI = transforms.Compose(
            [transforms.Resize(32),
             transforms.CenterCrop(32),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
             ]
    )

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             self.get_normalization_transform()
            ]
        )

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    transform, self.NAME)
        else:
            test_dataset = CIFAR10(base_path() + 'CIFAR10', train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    def get_clip_data_loaders(self):
        transform_clip = self.TRANSFORM_CLIP
        transform_xai = self.TRANSFORM_XAI

        train_dataset_clip = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform_clip)
        train_val_dataset_clip, test_val_dataset = get_train_data_clip(train_dataset_clip, transform_clip)

        train_dataset_xai = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform_xai)
        train_val_dataset_xai, test_val_dataset = get_train_data_clip(train_dataset_xai, transform_xai)

        print(self.i)
        val_train_clip = store_clip_masked_loaders(train_val_dataset_clip, self)
        val_train_xai = store_xai_masked_loaders(train_val_dataset_xai, self)

        return val_train_clip, val_train_xai

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    def increment_class(self):
        self.i += self.N_CLASSES_PER_TASK

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK
                        * SequentialCIFAR10.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform



