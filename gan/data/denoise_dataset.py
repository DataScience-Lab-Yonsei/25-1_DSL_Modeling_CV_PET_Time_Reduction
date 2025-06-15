import random
import os.path

import numpy as np
import nibabel as nib
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from data.base_dataset import BaseDataset


class DenoiseDataset(BaseDataset):
    """
    This dataset class can load nifti images.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        print('initializing datasets...')
        self.opt = opt
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.base_paths = sorted([os.path.join(self.dir_AB, x) for x in os.listdir(self.dir_AB)])

        self.A1_paths = [os.path.join(x, opt.input_nii_filename_A) for x in self.base_paths] ## Low Dose CT images
        self.B_paths = [os.path.join(x, opt.input_nii_filename_B) for x in self.base_paths]  ## High Dose CT images
        
        self.p_rotate = 0.75 if not self.opt.phase == 'train' else 0
        self.p_flip = 0.5 if not self.opt.phase == 'train' else 0

        self.paths = []
        
        for i, path in tqdm(enumerate(self.A1_paths), total=len(self.A1_paths)):
            self.paths += [(i, x) for x in range(nib.load(path).header['dim'][3])]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        
        if self.opt.phase == 'train':
            rotate = random.random() < self.p_rotate
            flip = random.random() < self.p_flip
            rotate = False
            flip = False
        A1_path = self.A1_paths[self.paths[index][0]]
        A1_index = self.paths[index][1]

        A1 = self.ct2tensor(A1_path, A1_index)
        if self.opt.input_nc == 1:
            A = A1

        elif self.opt.input_nc >= 3 and self.opt.input_nc % 2 == 1: # For odd
            k = (self.opt.input_nc - 1) // 2
            A_list = []
            for i in range(k, 0, -1):
                A0 = self.zeros() if A1_index - i < 0 else self.ct2tensor(A1_path, A1_index-i)
                A_list.append(A0)
            A_list.append(A1)
            M = nib.load(A1_path).header['dim'][3]
            for i in range(1, k+1):
                A0 = self.ct2tensor(A1_path, A1_index+i) if A1_index + i < M else self.zeros()
                A_list.append(A0)
            A = torch.cat(A_list, dim=0)

        if self.opt.phase != 'train':
            return {'A': A, 'B': A,
                    'A_paths': A1_path, 'B_paths': A1_path}
        
        B1_path = self.B_paths[self.paths[index][0]]
        B1_index = self.paths[index][1]

        B1 = self.ct2tensor(B1_path, B1_index)

        if self.opt.output_nc == 1:
            B = B1
        elif self.opt.model == 'cut':
                B = torch.cat([B1, B1, B1], dim=0)
        elif self.opt.output_nc >= 3 and self.opt.output_nc % 2 == 1:
            B_list = []
            k = (self.opt.output_nc - 1) // 2
            for i in range(k, 0, -1):
                B0 = self.zeros() if B1_index - i < 0 else self.ct2tensor(B1_path, B1_index-i)
                B_list.append(B0)
            # B_list.append(B1)
            B_list.append(B1)
            # B_list.append(B1)
            M = nib.load(B1_path).header['dim'][3]
            for i in range(1, k+1):
                B0 = self.ct2tensor(B1_path, B1_index+i) if B1_index + i < M else self.zeros()
                B_list.append(B0)
            B = torch.cat(B_list, dim=0)

        
        if self.opt.phase == 'train': # data augmentation
            if rotate:
                temp = random.random()
                if temp < 0.3333:
                    A = torch.rot90(A, 1, [1, 2])
                    B = torch.rot90(B, 1, [1, 2])   
                elif temp < 0.6667:
                    A = torch.rot90(A, 2, [1, 2])
                    B = torch.rot90(B, 2, [1, 2])
                else:
                    A = torch.rot90(A, 3, [1, 2])
                    B = torch.rot90(B, 3, [1, 2])   
                
            if flip:
                A = transforms.RandomVerticalFlip(1)(A)
                B = transforms.RandomVerticalFlip(1)(B)

        return {'A': A, 'B': B,
                'A_paths': A1_path, 'B_paths': B1_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """  
        return len(self.paths)

    def zeros(self):
        z = np.zeros((self.opt.load_size, self.opt.load_size))
        return transforms.ToTensor()(z).float()

    def ct2tensor(self, path, i):
        ct= nib.load(path).dataobj[..., i] # 1 slice
        ct = np.array(ct)
        ct = transforms.ToTensor()(ct)
        return transforms.Resize((self.opt.load_size, self.opt.load_size))(ct).float()