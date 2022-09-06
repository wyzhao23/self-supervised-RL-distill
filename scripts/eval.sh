CUDA_VISIBLE_DEVICES=0 xvfb-run python3 src/eval_kd.py \
    --domain_name cartpole \
    --task_name swingup \
    --action_repeat 8 \
    --mode color_hard \
    --use_inv \
    --num_shared_layers 8 \
    --seed 1 \
    --work_dir logs/cartpole_swingup/inv/1 \
    --pad_checkpoint 100k