import os
import argparse
import nibabel as nib
import numpy as np


def preprocess_nii(input_path, output_path):
    """Normalize a NIfTI file to the (0,1) range and save the result."""
    nii_img = nib.load(input_path)
    img_data = nii_img.get_fdata()

    if np.isnan(img_data).sum() == 0:
        min_val = np.min(img_data)
        max_val = np.max(img_data)

        if max_val - min_val > 1e-6:
            normalized_data = (img_data - min_val) / (max_val - min_val)
            print(f"[OK] {input_path} | min={min_val:.2f}, max={max_val:.2f}")
        else:
            print(f"[WARN] {input_path} | min ≈ max ({min_val}) → saving original data without normalization")
            normalized_data = img_data  # Save as-is
    else:
        print(f"[FAIL] NaN values detected in: {input_path}")
        return

    new_nii = nib.Nifti1Image(normalized_data, affine=nii_img.affine, header=nii_img.header)
    nib.save(new_nii, output_path)
    print(f"    → Saved: {output_path}")


def process_all_folders(root_dir):
    """Normalize specific NIfTI files in all subdirectories of the given root directory."""
    target_files = ["_FBB_golden.nii", "r_FBB_coreg.nii"]
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file in target_files:
                input_path = os.path.join(subdir, file)
                if file == "_FBB_golden.nii":
                    output_name = "processed_full_pet.nii"
                else:
                    output_name = "processed_partial_pet.nii"
                output_path = os.path.join(subdir, output_name)
                preprocess_nii(input_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Normalize PET NIfTI files to the (0,1) range.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing NIfTI files.")
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"[ERROR] Directory does not exist: {args.data_root}")
        return

    process_all_folders(args.data_root)


if __name__ == "__main__":
    main()
