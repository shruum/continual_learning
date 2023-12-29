import yaml
import os

job = yaml.load(open(r'template_transformer_cpu.yaml'))

best_params_seqcif10 = {
    200: {
        'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
    },
    500: {
        'idt': 'v1',
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
      },
}

best_params_seqcif100 = {
    200:{'idt': 'v1',
        'lr': 0.03,
        'minibatch_size': 32,
        'alpha_mm': [0.3],
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
        'aux': 'shape',
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
         },
   500: {'idt': 'v1',
        'lr': 0.1,
        'minibatch_size': 32,
        'alpha_mm': [0.3],
        'batch_size': 32,
        'n_epochs': 50,
        'img_size': 32,
        'aux': 'shape',
        'shape_filter': 'sobel',
        'shape_upsample_size': 64,
        'sobel_gauss_ksize': 3,
        'sobel_ksize': 3,
        'sobel_upsample': 'True',
         },
}

best_params_seqtiny = {
    200: {'lr': 0.03,
          'minibatch_size': 32,
          'softmax_temp': 2.0,
          'alpha': 0.1,
          'batch_size': 32,
          'n_epochs': 100,
          'aux': 'shape',
          'shape_filter': 'sobel',
          'shape_upsample_size': 128,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': 'True',
          'loss_type': ['kl'],
          },
    500: {'lr': 0.03,
          'minibatch_size': 32,
          'alpha': 0.1,
          'batch_size': 32,
          'n_epochs': 100,
          'aux': 'shape',
          'shape_filter': 'sobel',
          'shape_upsample_size': 128,
          'sobel_gauss_ksize': 3,
          'sobel_ksize': 3,
          'sobel_upsample': 'True',
          'loss_type': ['kl'],
          },
    5120: {'lr': 0.03,
           'minibatch_size': 32,
           'alpha': 0.1,
           'batch_size': 32,
           'n_epochs': 100,
           'aux': 'shape',
           'shape_filter': 'sobel',
           'shape_upsample_size': 128,
           'sobel_gauss_ksize': 3,
           'sobel_ksize': 3,
           'sobel_upsample': 'True',
           'loss_type': ['kl'],
           }
}


lst_buffer_size = [200, 500] #, 5120]
num_runs = 1
start_seed = 0
count = 0

datasets = ['seq-cifar10', 'seq-cifar100']

lst_datasets = [
    ('seq-cifar10', best_params_seqcif10),
    # ('seq-cifar100', best_params_seqcif100),
]

top_neurons = [10, 25, 30, 50]
layers = [("layer1", "layer2"), ("layer3", "layer4")]
grad_mult = [0.0, 0.1, 0.5]

for seed in range(start_seed, start_seed + num_runs):
    for dataset, params in lst_datasets:
        for buffer_size in lst_buffer_size:
                train_params = params[buffer_size]
                for l1, l2 in layers:
                    for n in top_neurons:
                        for mul in grad_mult:
                            exp_id = f"cl-diss-{buffer_size}-{dataset}-neur{n}--gradm{mul}-s{seed}"
                            job_args = ["-c", f"python /git/mammoth/main.py  \
                                --experiment_id {exp_id} \
                                --seed {seed} \
                                --model er \
                                --dataset {dataset} \
                                --buffer_size {buffer_size} \
                                --lr {train_params['lr']} \
                                --minibatch_size {train_params['minibatch_size']} \
                                --n_epochs {train_params['n_epochs']} \
                                --batch_size {train_params['batch_size']} \
                                --output_dir /output/er_xai \
                                --tensorboard \
                                --csv_log \
                                --top_neurons {n} \
                                --grad_multiplier {mul} \
                                --mask_layer {l1} {l2} \
                                "]
                        # set job params
                        job['metadata']['name'] = exp_id + '-shru'
                        job['spec']['template']['spec']['containers'][0]['args'] = job_args

                        yaml_out = 'temp/%s.yaml' % exp_id

                        with open(yaml_out, 'w') as outfile:
                            yaml.dump(job, outfile, default_flow_style=False)

                        count += 1
                        os.system('kubectl -n cyber-security-gpu create -f %s' % yaml_out)

print('%s jobs counted' % count)

#--aug_prob {p} \
