# Slice-Consistent 3D Volumetric Brain PET Time Reduction using GAN Models

### Requirements

```
conda env create -f environment.yml
conda activate GAN_PET
```

### System Dependencies

```bash
sudo apt update
sudo apt install -y libgl1-mesa-glx
```

### Dataset Preprocessing

We used [ADNI PET data](https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/pet/), and preprocessed it using the scripts in the `preprocessing/` directory .

### Dataset Structure
The directory containing PET data should follow this structure:
```commandline
/PET_dataset_root/
├── train/
│   ├── 006_S_4485/
│   │   ├── _FBB_golden.nii # 20min PET
│   │   └── r_FBB_coreg.nii # 4min PET
│   ├── 037_S_4214/
│   │   ├── _FBB_golden.nii
│   │   └── r_FBB_coreg.nii
│   ├── ...
├── val/
│   ├── ...
├── test/
│   ├── ...
```

### Training

For training with a custom PET dataset, use the following command:
```commandline
bash train.sh
```
To train a different model(e.g. CUT, CycleGAN) or use a custom configuration, edit the `train.sh` script accordingly. 

Available training options can be found in `options/train_options.py`.


### Testing

For testing with a custom PET dataset, use the following command:
```commandline
bash test.sh
```
After training, a config file (e.g., `train_opt.txt`) and model checkpoint (e.g., `latest_net_G.pth`) will be saved automatically.
You need to specify these files inside `test.sh` to ensure the test uses the correct settings and weights. 

Available test options can be found in `options/test_options.py`.


## Acknowledgement

This repository is heavily based on the official implementation of [DCLGAN](https://github.com/JunlinHan/DCLGAN).
