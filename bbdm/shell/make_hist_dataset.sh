for HW in 160
do
    partial_PET_name="processed_partial_pet.nii"
    full_PET_name="processed_full_pet.nii"

    for which in "train" "valid" "test"
    do
        # for plane in "axial" "sagittal" "coronal"
        for plane in "axial"
        do
            for hist_type in "normal" "avg" #"colin"
            do
            python -u ./brain_dataset_utils/generate_total_hist_global.py \
                    --plane $plane\
                    --hist_type $hist_type \
                    --which_set $which \
                    --height $HW \
                    --width $HW \
                    --pkl_name "./generated_datasets/Full_PET_hist_global_${HW}_${which}_${plane}_$hist_type.pkl" \
                    --data_dir "/root/ADNI_FBB_coreg2/ADNI_FBB_coreg2" \
                    --data_csv "/root/ADNI_FBB_coreg2/dataset.csv" \
                    --partial_PET_name $partial_PET_name \
                    --full_PET_name $full_PET_name \
                    > ./generated_datasets/hdf5_log/Full_PET_hist_global_${HW}_${which}_${plane}_$hist_type.log
            done      
        done      
    done
done      
