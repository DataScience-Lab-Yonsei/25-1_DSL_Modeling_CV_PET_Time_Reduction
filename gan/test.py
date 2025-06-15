import os
import ast

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torchvision.transforms as transforms

from setproctitle import setproctitle

from options.test_options import TestOptions
from models.networks import define_G


def load_test_options_from_txt(opt, txt_path):
    def safe_eval(value):
        try:
            return ast.literal_eval(value)
        except:
            return value

    with open(txt_path, 'r') as f:
        for line in f:
            if ':' in line and not line.startswith('-'):
                key, raw = line.strip().split(':', 1)
                value = raw.split('\t')[0].strip()

                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value == 'inf':
                    value = float('inf')
                else:
                    value = safe_eval(value)

                if key == 'nce_layers' and isinstance(value, tuple):
                    value = ','.join(map(str, value))

                if key == 'gpu_ids':
                    if isinstance(value, str):
                        value = [int(x) for x in value.split(',') if x.strip()]
                    elif isinstance(value, int):
                        value = [value]

                if hasattr(opt, key):
                    setattr(opt, key, value)

    opt.phase = 'test'
    opt.isTrain = False
    opt.eval = True

    return opt


def create_generator(opt, model_path):
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    state_dict = torch.load(model_path, map_location=device)
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    if opt.model == 'cycle_gan':
        if opt.direction == 'AtoB':
            model = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                   not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        elif opt.direction == 'BtoA':
            model = define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.normG,
                                   not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        else:
            raise ValueError(f"Invalid direction for CycleGAN: {opt.direction}. Must be 'AtoB' or 'BtoA'.")
    else:
        model = define_G(
            input_nc=opt.input_nc,
            output_nc=opt.output_nc,
            ngf=opt.ngf,
            netG=opt.netG,
            norm=opt.normG,
            use_dropout=(not opt.no_dropout),
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            no_antialias=opt.no_antialias,
            no_antialias_up=opt.no_antialias_up,
            gpu_ids=opt.gpu_ids,
            opt=opt
        )

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model


def denoise(opt, model, test_path, patient):
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    preprocess = True  
    scale = 2000
    
    nii = nib.load(os.path.join(test_path, patient, opt.input_nii_filename))
    img = np.array(nii.dataobj)
    
    orig = np.mean(img, axis=(0, 1))
    img = img / scale

    if preprocess:
        scale_factor = np.exp(np.sum(img * np.log(np.abs(img) + 1e-07), axis=(0, 1, 2)) / (np.sum(img, axis=(0, 1, 2))))
        img /= scale_factor

    outputs = []

    for i in range(0, nii.header['dim'][3]):
        if i == nii.header['dim'][3]-1:
            input = transforms.ToTensor()(img[..., i-1:i+1])
            input = transforms.Resize((160, 160))(input).float()
            a = np.zeros((160, 160))
            a = transforms.ToTensor()(a).float()
            input = torch.cat([input, a], dim=0)
        elif i == 0:
            input = transforms.ToTensor()(img[..., 0:2])
            input = transforms.Resize((160, 160))(input).float()
            a = np.zeros((160, 160))
            a = transforms.ToTensor()(a).float()
            input = torch.cat([a, input], dim=0)
        else:
            input = transforms.ToTensor()(img[..., i-1:i+2])
            input = transforms.Resize((160, 160))(input).float()

        input = torch.unsqueeze(input, dim=0).to(device)

        output = model(input)
        output = output.data[0].cpu().float().numpy()
        output = output[output.shape[0] // 2, ...]

        outputs.append(output)

    outputs = np.array(outputs).transpose(1, 2, 0)
    outputs *= scale

    if preprocess:
        outputs *= scale_factor

    shifted = np.mean(outputs, axis=(0, 1))
    outputs *= orig/shifted
    nii = type(nii)(outputs, affine=nii.affine, header=nii.header, extra=nii.extra, file_map=nii.file_map)

    output_path = os.path.join(opt.results_dir, f'{patient}{opt.output_filename_suffix}')
    print('created CT2MR image on', output_path)
    nib.save(nii, output_path)


if __name__ == '__main__':
    setproctitle('sb inference')
    
    opt = TestOptions().parse(verbose=False)
    opt = load_test_options_from_txt(opt, opt.train_opt_path)
    print(opt)
    os.makedirs(opt.results_dir, exist_ok=True)

    model = create_generator(opt, opt.checkpoint_path)
    test_path = os.path.join(opt.dataroot, opt.phase)
    data = os.listdir(test_path)

    for patient in tqdm(data):
        denoise(opt, model, test_path, patient)
