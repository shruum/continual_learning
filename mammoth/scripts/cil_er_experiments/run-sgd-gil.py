import yaml
import os

job = yaml.load(open(r'scripts/template.yaml'))

# lst_gil_seed = [15, 10, 5, 9, 18, 12, 16, 8, 1, 4]

#     500: [8, 12, 1993, 5, 7, 15, 19, 10, 11, 2],
#     1000: [12, 13, 1993, 15, 8, 3, 5, 10, 19, 2],

lst_gil_seed = [1993, 7, 15, 19, 11, 2, 13, 3]

lst_epoch = [100]
n_runs = 1
count = 0

lst_weight_dist = ['unif', 'longtail']


for gil_seed in lst_gil_seed:
    for run in range(n_runs):
        for n_epoch in lst_epoch:
            for weight_dist in lst_weight_dist:
                exp_id = f'fahad-sgd-gil-{weight_dist}-{gil_seed}-e-{n_epoch}-r-{run}'
                job_args = ["-c", f"python /git/continual_learning/mammoth/main.py  \
                    --experiment_id {'-'.join(exp_id.split('-')[1:])} \
                    --model sgd \
                    --dataset gil-cifar100 \
                    --gil_seed {gil_seed} \
                    --weight_dist {weight_dist} \
                    --batch_size 32 \
                    --lr 0.1 \
                    --n_epochs {n_epoch} \
                    --output_dir /output/sgd_gil_cifar100 \
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
