import yaml
import os

job = yaml.load(open(r'scripts/template.yaml'))

param_dict = {
    'seq-cifar10': {
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
    },

    'seq-tinyimg': {
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 100
    },

}

lst_dataset = ['seq-cifar10']

dict_dataset_alias = {
    'seq-cifar10': 'c10',
    'seq-tinyimg': 'tinyimg'
}

lst_stable_ema_param = [

    ('v1', 0.1, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v2', 0.2, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v3', 0.3, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v4', 0.4, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v5', 0.5, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v6', 0.6, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v7', 0.7, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v8', 0.8, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v9', 0.9, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),

]
lst_plastic_ema_param = [
    ('v1', 0.2, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v2', 0.3, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v3', 0.4, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v4', 0.5, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v5', 0.6, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v6', 0.7, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v7', 0.8, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v8', 0.9, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
    ('v9', 1.0, 0.999, 'constant', 78000, -0.65, 0.9, 0.999),
]

# sigmoid, phase_shift -5
lst_reg_weight_param = [
    # ('v1', 0.1, 'constant', 78000, -5, 0.01, 1),
    # ('v2', 0.15, 'constant', 78000, -5, 0.01, 1),
    ('v5', 0.2, 'constant', 78000, -5, 0.01, 1),
]

# lst_buffer_size = [200, 500, 5120]
lst_buffer_size = [500]
n_runs = 3
start_seed = 0
count = 0


for seed in range(start_seed, start_seed + n_runs):
    for dataset in lst_dataset:
        for buffer_size in lst_buffer_size:
            for reg_weight_id, reg_weight, reg_weight_rampup_fn, reg_weight_rampup_len, reg_weight_rampup_phase, reg_weight_min, reg_weight_max in lst_reg_weight_param:
                for stable_ema_id, stable_ema_update_freq, stable_ema, stable_ema_rampup_fn, stable_ema_rampup_len, stable_ema_rampup_phase, stable_ema_min, stable_ema_max in lst_stable_ema_param:
                    for plastic_ema_id, plastic_ema_update_freq, plastic_ema, plastic_ema_rampup_fn, plastic_ema_rampup_len, plastic_ema_rampup_phase, plastic_ema_min, plastic_ema_max in lst_plastic_ema_param:
                        exp_id = f'cls-er-{dict_dataset_alias[dataset]}-b-{buffer_size}-reg-{reg_weight_id}-s-{stable_ema_id}-p-{plastic_ema_id}-s-{seed}'
                        job_args = ["-c", f"python /git/continual_learning/mammoth/main.py  \
                            --experiment_id {'-'.join(exp_id.split('-')[1:])} \
                            --model dualmeaner_v6 \
                            --dataset {dataset} \
                            --buffer_size {buffer_size} \
                            --reg_weight {reg_weight} \
                            --reg_weight_rampup {reg_weight_rampup_fn} \
                            --reg_weight_rampup_length {reg_weight_rampup_len} \
                            --reg_weight_rampup_alpha {reg_weight_rampup_phase} \
                            --reg_weight_rampup_min {reg_weight_min} \
                            --reg_weight_rampup_max {reg_weight_max} \
                            --stable_ema_update_freq {stable_ema_update_freq} \
                            --stable_ema_alpha {stable_ema} \
                            --stable_ema_alpha_rampup {stable_ema_rampup_fn} \
                            --stable_ema_alpha_rampup_length {stable_ema_rampup_len} \
                            --stable_ema_alpha_rampup_alpha {stable_ema_rampup_phase} \
                            --stable_ema_alpha_rampup_min {stable_ema_min} \
                            --stable_ema_alpha_rampup_max {stable_ema_max} \
                            --plastic_ema_update_freq {plastic_ema_update_freq} \
                            --plastic_ema_alpha {plastic_ema} \
                            --plastic_ema_alpha_rampup {plastic_ema_rampup_fn} \
                            --plastic_ema_alpha_rampup_length {plastic_ema_rampup_len} \
                            --plastic_ema_alpha_rampup_alpha {plastic_ema_rampup_phase} \
                            --plastic_ema_alpha_rampup_min {plastic_ema_min} \
                            --plastic_ema_alpha_rampup_max {plastic_ema_max} \
                            --lr {param_dict[dataset]['lr']} \
                            --n_epochs {param_dict[dataset]['n_epochs']} \
                            --minibatch_size {param_dict[dataset]['minibatch_size']} \
                            --batch_size {param_dict[dataset]['batch_size']} \
                            --output_dir /output/cil_er_cifar10_sensitivity_analysis \
                            --tensorboard \
                            --csv_log \
                            "]
                        # set job params
                        job['metadata']['name'] = exp_id
                        job['spec']['template']['spec']['containers'][0]['args'] = job_args

                        yaml_out = 'scripts/temp/%s.yaml' % exp_id

                        with open(yaml_out, 'w') as outfile:
                            yaml.dump(job, outfile, default_flow_style=False)

                        if plastic_ema_update_freq > stable_ema_update_freq:
                            count += 1
                            os.system('kubectl -n arl create -f %s' % yaml_out)

print('%s jobs counted' % count)
