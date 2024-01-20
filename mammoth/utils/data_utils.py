import os
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from backbone.ResNet18 import resnet18
import matplotlib
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

DATASET_ROOTS = {"imagenet_val": "/data/input-ai/datasets/ImageNet100/val/",
                "broden": "data/broden1_224/images/",
                 "cifar100": "data/broden1_224/images/",
                 # "coco": "/data/input-ai/datasets/COCO/coco_classfn",
                "bdd": "/volumes1/safeexplain/clip_diss/CLIP-dissect/data/bdd",
                "eurocity":"/volumes1/safeexplain/clip_diss/CLIP-dissect/data/euro_city_person",
                "mapillary": "/volumes1/safeexplain/clip_diss/CLIP-dissect/data/mapillary"}


def mask_data(test_dataset, task, n_classes_per_task):
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    test_mask = np.logical_and(np.array(test_dataset.targets) >= task,
                               np.array(test_dataset.targets) < task + n_classes_per_task)

    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    return


def get_target_model(target_name, device, path=""):
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {resnet18_places, resnet18, resnet34, resnet50, resnet101, resnet152}
                 except for resnet18_places this will return a model trained on ImageNet from torchvision
                 
    To Dissect a different model implement its loading and preprocessing function here
    """
    if target_name == 'resnet18_places': 
        target_model = models.resnet18(num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    elif "vit_b" in target_name:
        target_name_cap = target_name.replace("vit_b", "ViT_B")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    elif "resnet18_cifar" in target_name:
        # from models.resnet import resnet18
        # target_model = resnet18(num_classes=100, pretrained=True)
        target_model = resnet18(nclasses=100)
        # path = '/data/output-ai/shruthi.gowda/continual/base_aug/results/class-il/seq-cifar100/derpp/derpp-base-aug-model-200-seq-cifar100-s1/net_0.pth'
        target_model = torch.load(path).to(device) #['state_dict']
        # target_model.load_state_dict(state_dict)
        target_model.eval()
        preprocess = get_resnet_cifar_preprocess()

    elif "resnet" in target_name:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    
    target_model.eval()
    return target_model, preprocess

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_resnet_cifar_preprocess():
    target_mean = [0.4914, 0.4822, 0.4465]
    target_std = [0.2023, 0.1994, 0.2010]
    SIZE = 32

    preprocess = transforms.Compose([transforms.RandomCrop(SIZE, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_data(dataset_name, preprocess=None, task=0):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar10_val":
        n_classes_per_task = 2
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)
        mask_data(data, task, n_classes_per_task)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])


    return data


def get_places_id_to_broden_label():
    with open("data/categories_places365.txt", "r") as f:
        places365_classes = f.read().split("\n")
    
    broden_scenes = pd.read_csv('data/broden1_224/c_scene.csv')
    id_to_broden_label = {}
    for i, cls in enumerate(places365_classes):
        name = cls[3:].split(' ')[0]
        name = name.replace('/', '-')
        
        found = (name+'-s' in broden_scenes['name'].values)
        
        if found:
            id_to_broden_label[i] = name.replace('-', '/')+'-s'
        if not found:
            id_to_broden_label[i] = None
    return id_to_broden_label
    
def get_cifar_superclass():
    cifar100_has_superclass = [i for i in range(7)]
    cifar100_has_superclass.extend([i for i in range(33, 69)])
    cifar100_has_superclass.append(70)
    cifar100_has_superclass.extend([i for i in range(72, 78)])
    cifar100_has_superclass.extend([101, 104, 110, 111, 113, 114])
    cifar100_has_superclass.extend([i for i in range(118, 126)])
    cifar100_has_superclass.extend([i for i in range(147, 151)])
    cifar100_has_superclass.extend([i for i in range(269, 281)])
    cifar100_has_superclass.extend([i for i in range(286, 298)])
    cifar100_has_superclass.extend([i for i in range(300, 308)])
    cifar100_has_superclass.extend([309, 314])
    cifar100_has_superclass.extend([i for i in range(321, 327)])
    cifar100_has_superclass.extend([i for i in range(330, 339)])
    cifar100_has_superclass.extend([345, 354, 355, 360, 361])
    cifar100_has_superclass.extend([i for i in range(385, 398)])
    cifar100_has_superclass.extend([409, 438, 440, 441, 455, 463, 466, 483, 487])
    cifar100_doesnt_have_superclass = [i for i in range(500) if (i not in cifar100_has_superclass)]
    
    return cifar100_has_superclass, cifar100_doesnt_have_superclass

def visualize(args, target_feats, similarities, words, pil_data, save_path):
    neurons_to_check = np.arange(0,64,1) #[10, 167, 234, 368, 450, 510]
    # torch.sort(torch.max(similarities, dim=1)[0], descending=True)[1][0:10]
    top_vals, top_ids = torch.topk(target_feats, k=5, dim=0)
    font_size = 14
    font = {'size': font_size}

    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=[10, (len(neurons_to_check)/2) * 2])  # constrained_layout=True)
    subfigs = fig.subfigures(nrows=len(neurons_to_check), ncols=1)

    for j, orig_id in enumerate(neurons_to_check):
        vals, ids = torch.topk(similarities[orig_id], k=similarities.size(1), largest=True)

        subfig = subfigs[j]
        subfig.text(0.13, 0.96, "Neuron {}:".format(int(orig_id)), size=font_size)
        subfig.text(0.27, 0.96, "CLIP-Dissect:", size=font_size)
        subfig.text(0.4, 0.96, words[int(ids[0])], size=font_size)
        axs = subfig.subplots(nrows=1, ncols=5)
        for i, top_id in enumerate(top_ids[:, orig_id]):
            im, label = pil_data[top_id]
            im = im.resize([64, 64])
            axs[i].imshow(im)
            axs[i].axis('off')

    plt.show()
    fig.savefig(os.path.join(save_path, '{}.png'.format(args.target_model)),
                bbox_inches='tight', dpi=500)