import subprocess
import glob
import os

def run_command(target_layer, concept_set, target_path, task, result_dir, similarity_fn):
    command = f"python3 visualize.py " \
              f"--clip_model {'ViT-B/16'} " \
              f"--target_model {'resnet18_cifar'} " \
              f"--target_layer {target_layer} " \
              f"--d_probe {'cifar10_val'} " \
              f"--concept_set {concept_set} " \
              f"--target_path {target_path} " \
              f"--task {task} " \
              f"--similarity_fn {similarity_fn} " \
              f"--result_dir {result_dir}"
    print(command)
    subprocess.call(command, shell=True)

tasks = [0, 1, 2, 3, 4]
models = ['net_t0_e49.pth', 'net_t1_e0.pth', 'net_t1_e49.pth', 'net_t2_e0.pth', 'net_t2_e49.pth',
          'net_t3_e0.pth', 'net_t3_e49.pth', 'net_t4_e0.pth', 'net_t4_e49.pth']
layers = ['layer1', 'layer2', 'layer3', 'layer4']

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

def get_task(model):
    if 't0' in model:
        task = 0
    elif 't1' in model:
        task = 1
    elif 't2' in model:
        task = 2
    elif 't3' in model:
        task = 3
    elif 't4' in model:
        task = 4
    return task

path = "/volumes2/continual_learning/mammoth/mammothssl/results/results/class-il/seq-cifar10/er/base_cif10_er_resl64" #
# path = "/volumes2/continual_learning/mammoth/mammothssl/results/results/class-il/seq-cifar10/derpp/base_cif10_derp_resl64/"
# path = "/volumes1/xai_cl/resnet_l64/sgd/sgd_results/sgd_cif10_resl64"
result_dir = "/volumes1/xai_cl/resnet_l64/er"
similarity_fns = ["soft_wpmi"] # ["soft_wpmi", "cos_similarity"]

for layer in layers:
    for model in models:
        target_path = os.path.join(path, model)
        task = get_task(model)
        print(model)
        print(task)
        concept_set = get_concept(task)
        for similarity_fn in similarity_fns:
            run_command(target_layer=layer,
                        concept_set=concept_set,
                        target_path=target_path,
                        task=task,
                        result_dir=result_dir,
                        similarity_fn=similarity_fn
                        )