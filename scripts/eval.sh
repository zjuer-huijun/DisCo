PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python WANDB_ENABLE=0 AZFUSE_USE_FUSE=0 NCCL_ASYNC_ERROR_HANDLING=0 \
python finetune_sdm_yaml.py \
--cf config/ref_attn_clip_combine_controlnet/tiktok_S256L16_xformers_tsv.py \
--eval_visu --root_dir /home/exp/disco/test \
--local_train_batch_size 32 \
--local_eval_batch_size 32 \
--log_dir exp/tiktok_ft --epochs 20 --deepspeed --eval_step 500 --save_step 500 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-4 --fix_dist_seed --loss_target "noise" \
--train_yaml /home/data/disco/tuning_data/composite_offset/train_TiktokDance-poses-masks.yaml \
--val_yaml /home/data/disco/tuning_data/composite_offset/new10val_TiktokDance-poses-masks.yaml \
--unet_unfreeze_type "all" \
--refer_sdvae \
--ref_null_caption False \
--combine_clip_local --combine_use_mask \
--conds "poses" "masks" \
--stage1_pretrain_path /home/models/disco/cfg_2/mp_rank_00_model_states.pt \
--drop_ref 0.05 \
--guidance_scale 1.5 \
--eval_save_filename /home/exp/disco/eval_vis >> log.txt 2>&1