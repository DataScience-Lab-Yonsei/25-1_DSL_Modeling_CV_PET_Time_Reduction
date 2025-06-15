# Slice-Consistent 3D Volumetric Brain PET Time Reduction with 2D Brownian Bridge Diffusion Model

### Requirements

```
conda env create -f environment.yml
conda activate SCOPE
```

### System Dependencies

```bash
sudo apt update
sudo apt install -y libgl1-mesa-glx
```

### Dataset Preprocessing

We used [ADNI PET data](https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/pet/), and preprocessed it using the scripts in the `preprocessing/` directory.

You also need `data_csv.csv` which should include following columns:
```csv
pid,set
```

- `pid`: Patient ID (e.g., 037_S_4214_2) that corresponds to the folder name or data instance

- `set`: One of ***train***, ***valid***, or ***test*** indicating the data split

Example snippet of data.csv:
```csv
pid,set
037_S_6032,train
116_S_6517_2,train
037_S_6977,valid
116_S_6750,test
```

### Dataset Structure
The directory containing PET data should follow this structure:
```commandline
/PET_dataset_root/
├── 037_S_6032/
│   ├── processed_full_pet.nii # 20min PET
│   └── processed_partial_pet.nii # 4min PET
├── 116_S_6517_2/
│   ├── processed_full_pet.nii
│   └── processed_partial_pet.nii
├── ...
```


### Dataset Preparation

For custom PET datasets, ensure to modify the `data_dir` and `data_csv` arguments in the `make_hdf5.sh` script to match your custom dataset paths:
```commandline
sh shell/make_hdf5.sh
```

To generate a histogram dataset (in .pkl format) for Style Key Conditioning (SKC) with a custom PET dataset, modify the `data_dir` and `data_csv` arguments in the `make_hist_dataset.sh` script to match your custom dataset paths:
```commandline
sh shell/make_hist_dataset.sh
```


### Training

For training with a custom PET dataset, use the following command:
```commandline
sh shell/train.sh
```
To resume training from a checkpoint, modify the `resume` related arguments in the `train.sh`.


### Testing

For testing with a custom PET dataset, use the following command:
```commandline
sh shell/test.sh
```


## Acknowledgement

This repository is heavily based on the official implementation of [CT2MRI](https://github.com/MICV-yonsei/CT2MRI).
