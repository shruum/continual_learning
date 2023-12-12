import yaml
import os

job = yaml.load(open(r'scripts/template.yaml'))

best_params = {
    200: {
        'idt': 'v1',
        'reg_weight': 0.15,
        'stable_ema_update_freq': 0.1,
        'stable_ema_alpha': 0.999,
        'plastic_ema_update_freq': 0.3,
        'plastic_ema_alpha': 0.999,
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
    },
    500: {
        'idt': 'v1',
        'reg_weight': 0.15,
        'stable_ema_update_freq': 0.1,
        'stable_ema_alpha': 0.999,
        'plastic_ema_update_freq': 0.9,
        'plastic_ema_alpha': 0.999,
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
      },
    5120: {
        'idt': 'v1',
        'reg_weight': 0.15,
        'stable_ema_update_freq': 0.8,
        'stable_ema_alpha': 0.999,
        'plastic_ema_update_freq': 1.0,
        'plastic_ema_alpha': 0.999,
        'lr': 0.1,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
      }
}

lst_buffer_size = [500, 200, 5120]
lst_ema_alpha = [0.999]
# lst_update_freq = {
#     200: [0.1, 0.2, 0.3],
#     500: [0.1, 0.3, 0.5, 0.7, 0.9],
#     5120: [0.8, 0.9, 1],
# }

lst_update_freq = {
    200: [0.2],
    500: [0.5],
    # 5120: [0.9],
    5120: [0.8],
}

num_runs = 9
start_seed = 1
count = 0

for seed in range(start_seed, start_seed + num_runs):
    for buffer_size in lst_buffer_size:
        train_params = best_params[buffer_size]

        for ema_alpha in lst_ema_alpha:
            for update_freq in lst_update_freq[buffer_size]:

                exp_id = f"ema-er-c10-{buffer_size}-param-{train_params['idt']}-{ema_alpha}-{update_freq}-s-{seed}"
                job_args = ["-c", f"python /git/continual_learning/mammoth/main.py  \
                    --experiment_id {'-'.join(exp_id.split('-')[1:])} \
                    --model emaer \
                    --dataset seq-cifar10 \
                    --buffer_size {buffer_size} \
                    --reg_weight {train_params['reg_weight']} \
                    --ema_update_freq {update_freq} \
                    --ema_alpha {ema_alpha} \
                    --lr {train_params['lr']} \
                    --n_epochs {train_params['n_epochs']} \
                    --minibatch_size {train_params['minibatch_size']} \
                    --batch_size {train_params['batch_size']} \
                    --output_dir /output/ema_er_cifar10 \
                    --tensorboard \
                    --csv_log \
                    "]
                # set job params
                job['metadata']['name'] = 'fahad-' + exp_id
                job['spec']['template']['spec']['containers'][0]['args'] = job_args

                yaml_out = 'scripts/temp/%s.yaml' % exp_id

                with open(yaml_out, 'w') as outfile:
                    yaml.dump(job, outfile, default_flow_style=False)

                count += 1
                os.system('kubectl -n arl create -f %s' % yaml_out)

print('%s jobs counted' % count)
