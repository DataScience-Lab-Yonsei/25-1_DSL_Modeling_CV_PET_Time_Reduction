#!/bin/bash
set -e

date="250604"

config_name="BBDM_base_ISTA_SKC.yaml"
HW="160"
plane="axial"
gpu_ids="0"
ddim_eta=0.0
prefix="Full_PET_global_hist_context"

exp_name="${date}_${HW}_${config_name%.*}_${plane}_DDIM_${prefix}"

# test
test_epoch="1"
resume_model="./results/${exp_name}/checkpoint/latest_model_${test_epoch}.pth"
resume_optim="./results/${exp_name}/checkpoint/latest_optim_sche_${test_epoch}.pth"
config_path="./results/${exp_name}/checkpoint/config_backup.yaml"

sample_step=100
inference_type="normal" # normal, average, ISTA_average, ISTA_mid
ISTA_step_size=0.5
num_ISTA_step=1

python ./main.py \
    --exp_name $exp_name \
    --config $config_path \
    --sample_to_eval \
    --gpu_ids $gpu_ids \
    --resume_model $resume_model \
    --resume_optim $resume_optim \
    --HW $HW \
    --plane $plane \
    --ddim_eta $ddim_eta \
    --sample_step $sample_step \
    --inference_type $inference_type \
    --ISTA_step_size $ISTA_step_size \
    --num_ISTA_step $num_ISTA_step


