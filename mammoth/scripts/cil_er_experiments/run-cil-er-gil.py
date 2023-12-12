import yaml
import os

job = yaml.load(open(r'scripts/template.yaml'))

best_params = {
    'unif': {
        200: {
            'idt': 'v1',
            'reg_weight': 0.15,
            'stable_ema_update_freq': 0.6,
            'stable_ema_alpha': 0.999,
            'plastic_ema_update_freq': 0.7,
            'plastic_ema_alpha': 0.999,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
        },
        500: {
            'idt': 'v1',
            'reg_weight': 0.15,
            'stable_ema_update_freq': 0.6,
            'stable_ema_alpha': 0.999,
            'plastic_ema_update_freq': 0.9,
            'plastic_ema_alpha': 0.999,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
          },
        1000: {
            'idt': 'v1',
            'reg_weight': 0.15,
            'stable_ema_update_freq': 0.6,
            'stable_ema_alpha': 0.999,
            'plastic_ema_update_freq': 0.8,
            'plastic_ema_alpha': 0.999,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
          },
    },
    'longtail': {
        200: {
            'idt': 'v1',
            'reg_weight': 0.1,
            'stable_ema_update_freq': 0.6,
            'stable_ema_alpha': 0.999,
            'plastic_ema_update_freq': 0.7,
            'plastic_ema_alpha': 0.999,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
        },
        500: {
            'idt': 'v1',
            'reg_weight': 0.1,
            'stable_ema_update_freq': 0.6,
            'stable_ema_alpha': 0.999,
            'plastic_ema_update_freq': 0.7,
            'plastic_ema_alpha': 0.999,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
        },
        1000: {
            'idt': 'v1',
            'reg_weight': 0.1,
            'stable_ema_update_freq': 0.6,
            'stable_ema_alpha': 0.999,
            'plastic_ema_update_freq': 0.8,
            'plastic_ema_alpha': 0.999,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
        },

    },
}

# lst_gil_seed = [1993,]
# lst_gil_seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1993]
lst_gil_seed = [13, 14, 15, 16, 17, 18, 19]
lst_buffer_size = [200, 500, 1000]
n_runs = 1
count = 0

lst_weight_dist = ['unif', 'longtail']
# lst_weight_dist = ['longtail']
for gil_seed in lst_gil_seed:
    for run in range(n_runs):
        for buffer_size in lst_buffer_size:
            for weight_dist in lst_weight_dist:
                exp_id = f"cil-er-gil-{weight_dist}-{gil_seed}-{buffer_size}-param-{best_params[weight_dist][buffer_size]['idt']}-s-{run}-c"
                job_args = ["-c", f"python /git/continual_learning/mammoth/main.py  \
                    --experiment_id {'-'.join(exp_id.split('-')[1:])} \
                    --model dualmeaner_v6 \
                    --dataset gil-cifar100 \
                    --gil_seed {gil_seed} \
                    --weight_dist {weight_dist} \
                    --buffer_size {buffer_size} \
                    --reg_weight {best_params[weight_dist][buffer_size]['reg_weight']} \
                    --reg_weight_rampup constant \
                    --stable_ema_update_freq {best_params[weight_dist][buffer_size]['stable_ema_update_freq']} \
                    --stable_ema_alpha {best_params[weight_dist][buffer_size]['stable_ema_alpha']} \
                    --stable_ema_alpha_rampup constant \
                    --plastic_ema_update_freq {best_params[weight_dist][buffer_size]['plastic_ema_update_freq']} \
                    --plastic_ema_alpha {best_params[weight_dist][buffer_size]['plastic_ema_alpha']} \
                    --plastic_ema_alpha_rampup constant \
                    --lr {best_params[weight_dist][buffer_size]['lr']} \
                    --n_epochs {best_params[weight_dist][buffer_size]['n_epochs']} \
                    --minibatch_size {best_params[weight_dist][buffer_size]['minibatch_size']} \
                    --batch_size {best_params[weight_dist][buffer_size]['batch_size']} \
                    --output_dir /output/cil_er_gil_cifar100_new \
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
