import yaml
import os

job = yaml.load(open(r'scripts/template.yaml'))

best_params = {
    200: {
        'idt': 'v2',
        'reg_weight': 0.1,
        'stable_ema_update_freq': 0.03,
        'stable_ema_alpha': 0.999,
        'plastic_ema_update_freq': 0.08,
        'plastic_ema_alpha': 0.999,
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
    },
    500: {
        'idt': 'v2',
        'reg_weight': 0.1,
        'stable_ema_update_freq': 0.03,
        'stable_ema_alpha': 0.999,
        'plastic_ema_update_freq': 0.08,
        'plastic_ema_alpha': 0.999,
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
      },
    5120: {
        'idt': 'v2',
        'reg_weight': 0.1,
        'stable_ema_update_freq': 0.05,
        'stable_ema_alpha': 0.999,
        'plastic_ema_update_freq': 0.08,
        'plastic_ema_alpha': 0.999,
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
      }
}

best_params = {
    200: {
        'idt': 'v3',
        'reg_weight': 0.1,
        'stable_ema_update_freq': 0.04,
        'stable_ema_alpha': 0.999,
        'plastic_ema_update_freq': 0.08,
        'plastic_ema_alpha': 0.999,
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
    },
    500: {
        'idt': 'v3',
        'reg_weight': 0.1,
        'stable_ema_update_freq': 0.05,
        'stable_ema_alpha': 0.999,
        'plastic_ema_update_freq': 0.08,
        'plastic_ema_alpha': 0.999,
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
      },
    5120: {
        'idt': 'v3',
        'reg_weight': 0.1,
        'stable_ema_update_freq': 0.07,
        'stable_ema_alpha': 0.999,
        'plastic_ema_update_freq': 0.08,
        'plastic_ema_alpha': 0.999,
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
      }
}

# lst_buffer_size = [500, 200, 5120]
lst_buffer_size = [5120]
num_runs = 10
start_seed = 15
count = 0

for seed in range(start_seed, start_seed + num_runs):
    for buffer_size in lst_buffer_size:

        train_params = best_params[buffer_size]

        exp_id = f"cls-er-tinyimg-{buffer_size}-param-{train_params['idt']}-s-{seed}"
        job_args = ["-c", f"python /git/continual_learning/mammoth/main.py  \
            --experiment_id {'-'.join(exp_id.split('-')[1:])} \
            --model clser \
            --dataset seq-tinyimg \
            --buffer_size {buffer_size} \
            --reg_weight {train_params['reg_weight']} \
            --stable_ema_update_freq {train_params['stable_ema_update_freq']} \
            --stable_ema_alpha {train_params['stable_ema_alpha']} \
            --plastic_ema_update_freq {train_params['plastic_ema_update_freq']} \
            --plastic_ema_alpha {train_params['plastic_ema_alpha']} \
            --lr {train_params['lr']} \
            --n_epochs {train_params['n_epochs']} \
            --minibatch_size {train_params['minibatch_size']} \
            --batch_size {train_params['batch_size']} \
            --output_dir /output/cls_er_imagenet_final \
            --tensorboard \
            --csv_log \
            --tiny_imgnet_path /output/datasets/ \
            "]
        # set job params
        job['metadata']['name'] = exp_id
        job['spec']['template']['spec']['containers'][0]['args'] = job_args

        yaml_out = 'scripts/temp/%s.yaml' % exp_id

        with open(yaml_out, 'w') as outfile:
            yaml.dump(job, outfile, default_flow_style=False)

        count += 1
        os.system('kubectl -n arl create -f %s' % yaml_out)

print('%s jobs counted' % count)
