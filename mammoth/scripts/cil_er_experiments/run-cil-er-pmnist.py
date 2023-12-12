import yaml
import os

job = yaml.load(open(r'scripts/template.yaml'))

best_params = {
    # 200: {
    #     'idt': 'v2',
    #     'reg_weight': 1.0,
    #     'stable_ema_update_freq': 0.8,
    #     'stable_ema_alpha': 0.99,
    #     'plastic_ema_update_freq': 1,
    #     'plastic_ema_alpha': 0.99,
    #     'lr': 0.2,
    #     'minibatch_size': 128,
    #     'batch_size': 128,
    #     'n_epochs': 1,
    # },
    200: {
        'idt': 'v3',
        'reg_weight': 1.0,
        'stable_ema_update_freq': 0.9,
        'stable_ema_alpha': 0.99,
        'plastic_ema_update_freq': 1,
        'plastic_ema_alpha': 0.99,
        'lr': 0.2,
        'minibatch_size': 128,
        'batch_size': 128,
        'n_epochs': 1,
    },
    500: {
        'idt': 'v3',
        'reg_weight': 1.0,
        'stable_ema_update_freq': 0.9,
        'stable_ema_alpha': 0.99,
        'plastic_ema_update_freq': 1,
        'plastic_ema_alpha': 0.99,
        'lr': 0.2,
        'minibatch_size': 128,
        'batch_size': 128,
        'n_epochs': 1,
      },
    5120: {
        'idt': 'v3',
        'reg_weight': 1.0,
        'stable_ema_update_freq': 0.95,
        'stable_ema_alpha': 0.99,
        'plastic_ema_update_freq': 1.0,
        'plastic_ema_alpha': 0.99,
        'lr': 0.2,
        'minibatch_size': 128,
        'batch_size': 128,
        'n_epochs': 1,
      }
}

lst_buffer_size = [500, 200, 5120]
num_runs = 25
start_seed = 0
count = 0

for seed in range(start_seed, start_seed + num_runs):
    for buffer_size in lst_buffer_size:

        train_params = best_params[buffer_size]

        exp_id = f"cil-er-pm-{buffer_size}-param-{train_params['idt']}-s-{seed}"
        job_args = ["-c", f"python /git/continual_learning/mammoth/main.py  \
            --experiment_id {'-'.join(exp_id.split('-')[1:])} \
            --model dualmeaner_v6 \
            --dataset perm-mnist \
            --buffer_size {buffer_size} \
            --reg_weight {train_params['reg_weight']} \
            --reg_weight_rampup constant \
            --stable_ema_update_freq {train_params['stable_ema_update_freq']} \
            --stable_ema_alpha {train_params['stable_ema_alpha']} \
            --stable_ema_alpha_rampup constant \
            --plastic_ema_update_freq {train_params['plastic_ema_update_freq']} \
            --plastic_ema_alpha {train_params['plastic_ema_alpha']} \
            --plastic_ema_alpha_rampup constant \
            --lr {train_params['lr']} \
            --n_epochs {train_params['n_epochs']} \
            --minibatch_size {train_params['minibatch_size']} \
            --batch_size {train_params['batch_size']} \
            --output_dir /output/cil_er_pmnist_final \
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
        os.system('kubectl -n arl delete -f %s' % yaml_out)

print('%s jobs counted' % count)
