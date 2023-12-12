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

lst_buffer_size = [500, 200, 5120]


lst_ema_alpha = [0.999]
# lst_update_freq = {
#     200: [0.04, 0.06, 0.08],
#     500: [0.04, 0.06, 0.08],
#     5120: [0.07, 0.08],
# }
lst_update_freq = {
    # 200: [0.04, 0.06, 0.08],
    200: [0.06],
    500: [0.08],
    5120: [0.08],
}


num_runs = 5
start_seed = 10
count = 0

for seed in range(start_seed, start_seed + num_runs):
    for buffer_size in lst_buffer_size:

        train_params = best_params[buffer_size]

        for ema_alpha in lst_ema_alpha:
            for update_freq in lst_update_freq[buffer_size]:

                exp_id = f"ema-er-tinyimg-{buffer_size}-param-{train_params['idt']}-{ema_alpha}-{update_freq}-s-{seed}"
                job_args = ["-c", f"python /git/continual_learning/mammoth/main.py  \
                    --experiment_id {'-'.join(exp_id.split('-')[1:])} \
                    --model emaer \
                    --dataset seq-tinyimg \
                    --buffer_size {buffer_size} \
                    --reg_weight {train_params['reg_weight']} \
                    --ema_update_freq {update_freq} \
                    --ema_alpha {ema_alpha} \
                    --lr {train_params['lr']} \
                    --n_epochs {train_params['n_epochs']} \
                    --minibatch_size {train_params['minibatch_size']} \
                    --batch_size {train_params['batch_size']} \
                    --output_dir /output/ema_er_imagenet_final \
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
