#!/bin/bash

gpu_number="0"

for model in dcl # "cut" # "cycle_gan" # "dcl"
do
    date=$(date '+%Y%m%d_%H%M%S')
    python -u train.py \
            --dataroot "/root/final/ADNI_FBB_coreg2" \
            --gpu_ids "${gpu_number}" \
            --model "${model}" \
            --model_name "${model}_${date}" \
            --dataset_mode "denoise" \
            --load_size 160 \
            --crop_size 160 \
            --input_nii_filename_A "r_FBB_coreg.nii" \
            --input_nii_filename_B "_FBB_golden.nii"
done
