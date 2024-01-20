from datasets.utils.continual_dataset import ContinualDataset
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

model = torch.load(r'/data/output/fahad.sarfraz/CLS-ER/cil_er_cifar10/saved_models/seq-cifar10/dualmeanerv6/er-c10-200-param-v1-s-6/model.ph')
stable_model = torch.load(r'/data/output/fahad.sarfraz/CLS-ER/cil_er_cifar10/saved_models/seq-cifar10/dualmeanerv6/er-c10-200-param-v1-s-6/stable_ema_model.ph')
plastic_model = torch.load(r'/data/output/fahad.sarfraz/CLS-ER/cil_er_cifar10/saved_models/seq-cifar10/dualmeanerv6/er-c10-200-param-v1-s-6/plastic_ema_model.ph')

data = np.load(r'/data/output/fahad.sarfraz/CLS-ER/cil_er_cifar10/saved_models/seq-cifar10/dualmeanerv6/er-c10-200-param-v1-s-6/buffer.npz')

images, labels = data['examples'], data['labels']

dataset = TensorDataset(torch.Tensor(images), torch.Tensor(labels))
dataloader = DataLoader(dataset, batch_size=32)

data_iter = iter(dataloader)


data, labels = next(data_iter)

unique_classes, counts = np.unique(labels, return_counts=True)
sort_idx = np.argsort(unique_classes)
unique_classes = unique_classes[sort_idx]
counts = counts[sort_idx]

n_unique_classes = len(unique_classes)

feats = torch.randn(32, 512, 4, 4)

class_prototypes = torch.zeros([n_unique_classes] + list(feats.shape)[1:])

for sample_idx in range(feats.shape[0]):
    sample_label = int(labels[sample_idx])
    class_prototypes[sample_label] += feats[sample_idx]


counts = torch.Tensor(counts)
counts = counts.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
class_prototypes /= counts

# Create a vector with class-wise accuracies
stable_model_feats, stable_model_logits = stable_model.extract_features(data.cuda())
plastic_model_feats, plastic_model_logits = plastic_model.extract_features(data.cuda())



confusion_matrix = torch.zeros(n_unique_classes, n_unique_classes)
_, stable_model_preds = torch.max(stable_model_logits, 1)
_, plastic_model_preds = torch.max(plastic_model_logits, 1)


def get_classwise_acc(output, target, num_classes):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    _, preds = torch.max(output, 1)

    for t, p in zip(target.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    classwise_acc = confusion_matrix.diag()/confusion_matrix.sum(1)

    return classwise_acc


stable_classwise = get_classwise_acc(stable_model_logits, labels, n_unique_classes)
plastic_classwise = get_classwise_acc(plastic_model_logits, labels, n_unique_classes)

stable_class_prototypes = class_prototypes
plastic_class_prototypes = class_prototypes

feat_sel_idx = stable_classwise > plastic_classwise

ema_feats = torch.where(
    feat_sel_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
    stable_class_prototypes,
    plastic_class_prototypes,
)

# =============================================================================
# MixUp
# =============================================================================
from torch.distributions.beta import Beta
import torch.nn.functional as F

minibatch_size = data.shape[0]
lambda_dist = Beta(1, 1)

data, labels = data.cuda(), labels.cuda()

lam = lambda_dist.sample(labels.shape).to(labels.device)

lam = lambda_dist.sample()

mixup_index = torch.randperm(minibatch_size).to(labels.device)
buf_mixup_inputs = lam * data + (1 - lam) * data[mixup_index]

stable_model_feats, stable_model_logits = stable_model.extract_features(buf_mixup_inputs)
plastic_model_feats, plastic_model_logits = plastic_model.extract_features(buf_mixup_inputs)


stable_model_prob = F.softmax(stable_model_logits, dim=1)
plastic_model_prob = F.softmax(plastic_model_logits, dim=1)

labels1 = labels[mixup_index]

mixup_labels = lam * F.one_hot(labels.long(), num_classes=10) + (1 - lam) * F.one_hot(labels.long()[mixup_index], num_classes=10)

lst_sel_idx = []
lst_stable_l2 = []
lst_plastic_l2 = []
for sample_idx in range(minibatch_size):

    indices = [int(labels[sample_idx].item()), int(labels1[sample_idx].item())]
    l2_stable = F.mse_loss(mixup_labels[sample_idx][indices], stable_model_prob[sample_idx][indices])
    l2_plastic = F.mse_loss(mixup_labels[sample_idx][indices], plastic_model_prob[sample_idx][indices])

    lst_stable_l2.append(l2_stable)
    lst_plastic_l2.append(l2_plastic)

    lst_sel_idx.append(l2_stable < l2_plastic)

sel_idx = torch.Tensor(lst_sel_idx).bool().cuda()

mixup_logits = torch.where(
    sel_idx.unsqueeze(-1),
    stable_model_logits,
    plastic_model_logits,
)




list(zip(labels, labels[mixup_index]))




one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=10)
