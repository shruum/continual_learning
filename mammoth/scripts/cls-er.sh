python main.py  \
--experiment_id test \
--model clser \
--dataset perm-mnist \
--reg_weight 1.0 \
--stable_ema_alpha 0.99 \
--plastic_ema_alpha 0.99 \
--stable_ema_update_freq 0.8 \
--stable_ema_update_freq 0.9 \
--batch_size 32 \
--buffer_size 200 \
--minibatch_size 32 \
--lr 0.1 \
--n_epochs 100 \
--output_dir sgd \
--tensorboard \
--csv_log \

