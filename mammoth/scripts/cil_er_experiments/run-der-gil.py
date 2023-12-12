import yaml
import os

job = yaml.load(open(r'scripts/template.yaml'))


best_params = {
    'unif': {
        200: {
            'idt': 'v1',
            'alpha': 0.2,
            'beta': 0.5,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
            'seeds': [1, 3]
        },
        500: {
            'idt': 'v1',
            'alpha': 0.2,
            'beta': 0.6,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
            'seeds': [1, 4]
        },
        1000: {
            'idt': 'v1',
            'alpha': 0.3,
            'beta': 0.6,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
            'seeds': []
        },
    },
    'longtail': {
        200: {
            'idt': 'v1',
            'alpha': 0.2,
            'beta': 0.6,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
            'seeds': [1, 3]
        },
        500: {
            'idt': 'v1',
            'alpha': 0.2,
            'beta': 0.8,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
            'seeds': []
        },
        1000: {
            'idt': 'v1',
            'alpha': 0.3,
            'beta': 0.9,
            'lr': 0.1,
            'minibatch_size': 32,
            'batch_size': 32,
            'n_epochs': 100,
            'seeds': [2, 4]
        },
    },
}


# lst_gil_seed = [1993,]
# lst_gil_seed = [0, 1, 2, 3, 4]
# lst_gil_seed = [5, 6, 7, 8, 9, 10, 11, 12]
lst_gil_seed = [13, 14, 15, 16, 17, 18, 19]
lst_buffer_size = [200, 500, 1000]
n_runs = 1
count = 0

lst_weight_dist = ['unif', 'longtail']

for gil_seed in lst_gil_seed:
    for run in range(n_runs):
        for buffer_size in lst_buffer_size:
            for weight_dist in lst_weight_dist:
                exp_id = f"fahad-der-gil-{weight_dist}-{gil_seed}-b-{buffer_size}-param-{best_params[weight_dist][buffer_size]['idt']}-r-{run}"
                job_args = ["-c", f"python /git/continual_learning/mammoth/main.py  \
                    --experiment_id {'-'.join(exp_id.split('-')[1:])} \
                    --model derpp \
                    --dataset gil-cifar100 \
                    --gil_seed {gil_seed} \
                    --weight_dist {weight_dist} \
                    --buffer_size {buffer_size} \
                    --alpha {best_params[weight_dist][buffer_size]['alpha']} \
                    --beta {best_params[weight_dist][buffer_size]['beta']} \
                    --minibatch_size {best_params[weight_dist][buffer_size]['minibatch_size']} \
                    --batch_size {best_params[weight_dist][buffer_size]['batch_size']} \
                    --lr {best_params[weight_dist][buffer_size]['lr']} \
                    --n_epochs {best_params[weight_dist][buffer_size]['n_epochs']} \
                    --output_dir /output/er_gil_cifar100 \
                    --tensorboard \
                    --csv_log \
                    "]
                # set job params
                job['metadata']['name'] = exp_id
                job['spec']['template']['spec']['containers'][0]['args'] = job_args

                yaml_out = 'scripts/temp/%s.yaml' % exp_id

                with open(yaml_out, 'w') as outfile:
                    yaml.dump(job, outfile, default_flow_style=False)

                # if gil_seed in best_params[weight_dist][buffer_size]['seeds']:
                count += 1
                os.system('kubectl -n arl create -f %s' % yaml_out)

print('%s jobs counted' % count)
