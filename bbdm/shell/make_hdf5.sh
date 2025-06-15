mkdir -p generated_datasets
mkdir -p generated_datasets/hdf5_log

for HW in 160 
do
    partial_PET_name="processed_partial_pet.nii"
    full_PET_name="processed_full_pet.nii"

    for which in "train" "valid" "test"
    do
        # for plane in "axial" "sagittal" "coronal"
        for plane in "axial"
        do
        python -u ./brain_dataset_utils/generate_total_hdf5_csv.py \
                --plane  $plane\
                --which_set $which \
                --height $HW \
                --width $HW \
                --hdf5_name "./generated_datasets/${HW}_${which}_${plane}.hdf5" \
                --data_dir "/root/ADNI_FBB_coreg2/ADNI_FBB_coreg2" \
                --data_csv "/root/ADNI_FBB_coreg2/dataset.csv" \
                --partial_PET_name $partial_PET_name \
                --full_PET_name $full_PET_name \
                > ./generated_datasets/hdf5_log/${HW}_${which}_${plane}.log
        done      
    done
done      
