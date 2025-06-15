#!/bin/bash
set -e

date="250604"

config_name="BBDM_base_ISTA_SKC.yaml"
HW="160"
plane="axial"
gpu_ids="0"
batch=4
ddim_eta=0.0
prefix="Full_PET_global_hist_context"

exp_name="${date}_${HW}_${config_name%.*}_${plane}_DDIM_${prefix}"
dataset_type="ct2mr_aligned_global_hist_context"

mkdir -p ./results/${exp_name}

resume=false
#resume=true
#resume_model="./results/${exp_name}/checkpoint/last_model.pth"
#resume_optim="./results/${exp_name}/checkpoint/last_optim_sche.pth"

cmd="python -u ./main.py \
    --train \
    --exp_name $exp_name \
    --config ./configs/$config_name \
    --dataset_type $dataset_type \
    --HW $HW \
    --plane $plane \
    --batch $batch \
    --ddim_eta $ddim_eta \
    --sample_at_start \
    --save_top \
    --gpu_ids $gpu_ids"
    

if [ "$resume" = true ]; then
    cmd="$cmd \
    --resume_model $resume_model \
    --resume_optim $resume_optim"
fi

eval $cmd