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

lst_reg_weight = [0]
# lst_at_weight = [0, 0.5, 1.0]
lst_at_weight = [0]
lst_prot_weight = [0.1, 0.5, 1.0]
lst_prot_level = ['batch']
lst_feat_level = [3]
lst_buffer_size = [500]
num_runs = 3
start_seed = 0
count = 0

for seed in range(start_seed, start_seed + num_runs):
    for buffer_size in lst_buffer_size:
        for feat_level in lst_feat_level:
            for reg_weight in lst_reg_weight:
                for at_weight in lst_at_weight:
                    for prot_weight in lst_prot_weight:
                        for prot_level in lst_prot_level:

                            train_params = best_params[buffer_size]

                            exp_id = f"clserpp-c10-{buffer_size}-param-{train_params['idt']}-r-{reg_weight}-f-{feat_level}-at-{at_weight}-prot-{prot_level}-{prot_weight}-s-{seed}"
                            job_args = ["-c", f"python /git/continual_learning/mammoth/main.py  \
                                --experiment_id {'-'.join(exp_id.split('-')[1:])} \
                                --model clserpp \
                                --dataset seq-cifar10 \
                                --buffer_size {buffer_size} \
                                --reg_weight {reg_weight} \
                                --at_weight {at_weight} \
                                --prot_weight {prot_weight} \
                                --prot_level {prot_level} \
                                --feat_level {feat_level} \
                                --stable_model_update_freq {train_params['stable_ema_update_freq']} \
                                --stable_model_alpha {train_params['stable_ema_alpha']} \
                                --plastic_model_update_freq {train_params['plastic_ema_update_freq']} \
                                --plastic_model_alpha {train_params['plastic_ema_alpha']} \
                                --lr {train_params['lr']} \
                                --n_epochs {train_params['n_epochs']} \
                                --minibatch_size {train_params['minibatch_size']} \
                                --batch_size {train_params['batch_size']} \
                                --output_dir /output/cil_er_pp_cifar10 \
                                --tensorboard \
                                --csv_log \
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
