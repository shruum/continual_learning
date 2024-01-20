python main.py  \
--model clsermixup \
--dataset seq-cifar10 \
--reg_weight 0.15 \
--mixup_alpha 4 \
--feat_level 3 \
--stable_model_alpha 0.99 \
--plastic_model_alpha 0.99 \
--stable_model_update_freq 0.8 \
--stable_model_update_freq 0.9 \
--batch_size 32 \
--buffer_size 200 \
--minibatch_size 32 \
--lr 0.1 \
--n_epochs 5 \
--tensorboard \
--csv_log \
--experiment_id v1 \

