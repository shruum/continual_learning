python main.py  \
--experiment_id test_ema \

--output_dir tiny_imagenet \
--model emaer \
--dataset perm-mnist \
--reg_weight 1.0 \
--ema_alpha 0.99 \
--ema_update_freq 0.8 \
--batch_size 32 \
--buffer_size 200 \
--minibatch_size 32 \
--lr 0.1 \
--n_epochs 5 \
--tensorboard \
--csv_log \

