import yaml
import os

job = yaml.load(open(r'scripts/template.yaml'))


# lst_gil_seed = [1993,]
# lst_gil_seed = [0, 1, 2, 3, 4]
# lst_gil_seed = [5, 6, 7, 8, 9, 10, 11, 12]
lst_gil_seed = [13, 14, 15, 16, 17, 18, 19]
lst_buffer_size = [200, 500, 1000]

lst_epoch = [100]
n_runs = 1
count = 0

lst_weight_dist = ['unif', 'longtail']

for gil_seed in lst_gil_seed:
    for run in range(n_runs):
        for n_epoch in lst_epoch:
            for buffer_size in lst_buffer_size:
                for weight_dist in lst_weight_dist:
                    exp_id = f'fahad-er-gil-{weight_dist}-{gil_seed}-b-{buffer_size}-e-{n_epoch}-r-{run}'
                    job_args = ["-c", f"python /git/continual_learning/mammoth/main.py  \
                        --experiment_id {'-'.join(exp_id.split('-')[1:])} \
                        --model er \
                        --dataset gil-cifar100 \
                        --gil_seed {gil_seed} \
                        --weight_dist {weight_dist} \
                        --buffer_size {buffer_size} \
                        --minibatch_size 32 \
                        --batch_size 32 \
                        --lr 0.1 \
                        --n_epochs {n_epoch} \
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

                    count += 1
                    os.system('kubectl -n arl create -f %s' % yaml_out)

print('%s jobs counted' % count)
