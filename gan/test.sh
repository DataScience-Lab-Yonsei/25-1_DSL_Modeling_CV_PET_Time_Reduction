#!/bin/bash

gpu_number="0"

# Modify follwing paths
train_opt_path="./checkpoints/dcl_20250610_155913/train_opt.txt"
checkpoint_path="./checkpoints/dcl_20250610_155913/latest_net_G_A.pth"

python -u test.py \
        --dataroot "/root/final/ADNI_FBB_coreg2" \
        --gpu_ids "${gpu_number}" \
        --train_opt_path $train_opt_path \
        --checkpoint_path $checkpoint_path \
        --input_nii_filename "r_FBB_coreg.nii" \
        --output_filename_suffix "_gen.nii"
