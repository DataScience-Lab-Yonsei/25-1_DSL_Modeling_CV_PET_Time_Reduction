import os
import argparse

import numpy as np
from scipy.ndimage import affine_transform
import ants
from tqdm import tqdm
import nibabel as nib
from nibabel.freesurfer.mghformat import MGHHeader


def map_image(img, out_affine, out_shape, ras2ras=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
              order=1):
    """
    Function to map image to new voxel space (RAS orientation)

    :param nibabel.MGHImage img: the src 3D image with data and affine set
    :param np.ndarray out_affine: trg image affine
    :param np.ndarray out_shape: the trg shape information
    :param np.ndarray ras2ras: ras2ras an additional maping that should be applied (default=id to just reslice)
    :param int order: order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: np.ndarray new_data: mapped image data array
    """


    # compute vox2vox from src to trg
    vox2vox = np.linalg.inv(out_affine) @ ras2ras @ img.affine

    # here we apply the inverse vox2vox (to pull back the src info to the target image)
    image_data = np.asanyarray(img.dataobj)
    # convert frames to single image
    if len(image_data.shape) > 3:
        if any(s != 1 for s in image_data.shape[3:]):
            raise ValueError(f'Multiple input frames {tuple(image_data.shape)} not supported!')
        image_data = np.squeeze(image_data, axis=tuple(range(3,len(image_data.shape))))

    new_data = affine_transform(image_data, np.linalg.inv(vox2vox), output_shape=out_shape, order=order)
    return new_data, vox2vox


def getscale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    """
    Function to get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.
    Equivalent to how mri_convert conforms images.

    :param np.ndarray data: image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param f_low: robust cropping at low end (0.0 no cropping)
    :param f_high: robust cropping at higher end (0.999 crop one thousandths of high intensity voxels)
    :return: float src_min: (adjusted) offset
    :return: float scale: scale factor
    """
    data -= data.min()

    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cummulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # print("bin min: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(cs[idx])+"\n")
    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print('ERROR: rescale upper bound not found')

    src_max = idx * bin_size + src_min
    # print("bin max: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(voxnum-cs[idx])+"\n")

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))

    return src_min, scale


def scalecrop(data, dst_min, dst_max, src_min, scale):
    """
    Function to crop the intensity ranges to specific min and max values

    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param float src_min: minimal value to consider from source (crops below)
    :param float scale: scale value by which source will be shifted
    :return: np.ndarray data_new: scaled image data
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    print("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))

    return data_new


def rescale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    """
    Function to rescale image intensity values (0-255)

    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param f_low: robust cropping at low end (0.0 no cropping)
    :param f_high: robust cropping at higher end (0.999 crop one thousandths of high intensity voxels)
    :return: np.ndarray data_new: scaled image data
    """
    src_min, scale = getscale(data, dst_min, dst_max, f_low, f_high)
    data_new = scalecrop(data, dst_min, dst_max, src_min, scale)
    return data_new


def conform_PET(source_path, target_path, save_files, order=1):
    """
    Python version of mri_convert -c, modified to conform images to 160x160x96 dimensions
    with a 1.5 mm isotropic voxel size.

    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage new_img: conformed image
    """

    
    img = nib.load(source_path)
    cwidth, cheight, cdepth = 160, 160, 96
    csize = 1.5
    
    h1 = MGHHeader.from_header(img.header)  # Copy some parameters if input was MGH format
    h1.set_data_shape([cwidth, cheight, cdepth, 1])
    h1.set_zooms([csize, csize, csize])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = cwidth * csize
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
    
    # Map image data to new conformed space
    mapped_data, orig_to_new = map_image(img, h1.get_best_affine(), h1.get_data_shape(), order=order)
    
    new_img = nib.MGHImage(mapped_data, h1.get_best_affine(), h1)
    if save_files:
        nib.save(new_img, target_path)
        print("Write file :", target_path)
    
    return type(img)(mapped_data, affine=h1.get_best_affine(), header=h1)


def main():
    parser = argparse.ArgumentParser(description="Conform and register PET images using ANTs")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing subject folders')
    args = parser.parse_args()

    root_dir = args.data_root

    # Get the list of patient IDs (subdirectories)
    pids = [pid for pid in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, pid))]

    # Loop through each patient ID folder
    for _, pid in tqdm(enumerate(pids), total=len(pids)):
        # Define the base directory for the current patient
        FILE_DIR = os.path.join(root_dir, pid)

        # Define file paths for PET, MRI, and others
        file_FBB = os.path.join(FILE_DIR, 'FBB.nii')
        file_ST_256 = os.path.join(FILE_DIR, 'MRI_orig.nii')
        file_PET_conformed = os.path.join(FILE_DIR, 'FBB_conformed.nii')
        file_PET_resliced = os.path.join(FILE_DIR, 'r_FBB.nii')

        # Step 1: Conform the PET image if necessary
        if os.path.exists(file_FBB) and not os.path.exists(file_PET_conformed):
            conform_PET(file_FBB, file_PET_conformed, save_files=True)
            
        # Step 2: Perform registration if the resliced PET image does not exist
        if os.path.exists(file_ST_256) and os.path.exists(file_PET_conformed) and not os.path.exists(file_PET_resliced):
            src_image = ants.image_read(file_PET_conformed)
            dst_image = ants.image_read(file_ST_256)
            
            # Perform rigid registration
            registration = ants.registration(fixed=dst_image, moving=src_image, type_of_transform='Rigid')
            
            # Save the resliced PET image
            ants.image_write(registration['warpedmovout'], file_PET_resliced)
            print(f"Resliced PET image saved to: {file_PET_resliced}")


if __name__ == "__main__":
    main()