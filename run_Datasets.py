import sys
import os
import shutil
import numpy as np
import random

import torch.hub
# from Augmentations.augmentations import Augment_DRR
from numba import prange
import imgaug
from tqdm import tqdm
from SubXR_configs_parser import SubXRParser
from utils import dir_content, size_dir_content
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from Models.SR_GAN_big.model import Generator

remove_and_create = lambda x: (not shutil.rmtree(x, ignore_errors=True)) and os.makedirs(x)
# def create_datasets(data_path,datasets_path,f_name, j, train=True):
#     seed_n = random.randint(0, 2 ** 32 - 1)
#
#     # xr2ulna
#     shutil.copy(os.path.join(data_path, 'DRR', 'Input', f_name),
#                 os.path.join(datasets_path, 'xr2ulna', 'trainA' if train else 'testA'))
#     Augment_DRR(os.path.join(datasets_path, 'xr2ulna', 'trainA' if train else 'testA'), f_name,j,seed_n)
#     os.remove(os.path.join(datasets_path, 'xr2ulna', 'trainA' if train else 'testA', f_name))
#
#     shutil.copy(os.path.join(data_path, 'DRR', 'Ulna', f_name),
#                 os.path.join(datasets_path, 'xr2ulna', 'trainB' if train else 'testB'))
#     Augment_DRR(os.path.join(datasets_path, 'xr2ulna', 'trainB' if train else 'testB'), f_name,j,seed_n)
#     os.remove(os.path.join(datasets_path, 'xr2ulna', 'trainB' if train else 'testB', f_name))
#
#     # xr2radius
#     shutil.copy(os.path.join(data_path, 'DRR', 'Input', f_name),
#                 os.path.join(datasets_path, 'xr2radius', 'trainA' if train else 'testA'))
#     Augment_DRR(os.path.join(datasets_path, 'xr2radius', 'trainA' if train else 'testA'), f_name,j,seed_n)
#     os.remove(os.path.join(datasets_path, 'xr2radius', 'trainA' if train else 'testA',f_name))
#
#     shutil.copy(os.path.join(data_path, 'DRR', 'Radius', f_name),
#                 os.path.join(datasets_path, 'xr2radius', 'trainB' if train else 'testB'))
#     Augment_DRR(os.path.join(datasets_path, 'xr2radius', 'trainB' if train else 'testB'), f_name,j,seed_n)
#     os.remove(os.path.join(datasets_path, 'xr2radius', 'trainB' if train else 'testB', f_name))
#
#
#     # xr2ulna_n_radius
#     shutil.copy(os.path.join(data_path, 'DRR', 'Input', f_name),
#                 os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainA' if train else 'testA'))
#     Augment_DRR(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainA' if train else 'testA'), f_name,j,seed_n)
#     os.remove(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainA' if train else 'testA', f_name))
#
#     shutil.copy(os.path.join(data_path, 'DRR', 'Ulna', f_name),
#                 os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB1' if train else 'testB1'))
#     Augment_DRR(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB1' if train else 'testB1'), f_name,j,seed_n)
#     os.remove(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB1' if train else 'testB1', f_name))
#
#     shutil.copy(os.path.join(data_path, 'DRR', 'Radius', f_name),
#                 os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB2' if train else 'testB2'))
#     Augment_DRR(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB2' if train else 'testB2'), f_name,j,seed_n)
#     os.remove(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB2' if train else 'testB2', f_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
LR = 400
HR = 800
TR = 700
TEST = 0.0
SR_GAN = Generator().to(device)

def downsample_transform(im, LR=400):
    LR_im = cv2.resize(im, (LR, LR))
    return LR_im

def upsample_transform(im, HR=700):
    LR_im = cv2.resize(im, (HR, HR))
    return LR_im

def numpy_2_torch_transform(im):
    torch_im = (torch.from_numpy(np.transpose(im, [2, 0, 1])[None]) / 255.).half().to(device)

    return torch_im

def torch_2_numpy_transform(im):
    np_im = np.transpose(im.float().cpu().detach().numpy(), [1, 2, 0])*255.
    return np_im

def super_resolution_transform(im, SR_GAN_path, LR=400, HR=800, TR=700):
    SR_GAN.load_state_dict(torch.load(SR_GAN_path, map_location=device))
    SR_GAN.eval()
    SR_GAN.half()
    if im.shape[1]<TR:
        LR_im = downsample_transform(im, LR)
        torch_LR_im = numpy_2_torch_transform(LR_im)
        torch_HR_im = SR_GAN(torch_LR_im)
        HR_im = torch_2_numpy_transform(torch_HR_im)
    else:
        HR_im = upsample_transform(im, HR)
    return HR_im

TRANSFORMS = {
    "down_sample": downsample_transform,
    "up_sample":   upsample_transform,
    "SR_GAN":      super_resolution_transform
}

def parse_transform(transform):
    for k, args in transform.items():
        if k in TRANSFORMS:
            return lambda x :TRANSFORMS[k](x, *args)
    return lambda x: x

def create_datasets(configs, dataset_type):
    remove_and_create(configs['Datasets'][dataset_type]['out_dir'])
    for suffix in dir_content(configs['Datasets'][dataset_type]['out_dir'], random=False):
        remove_and_create(os.path.join(configs['Datasets'][dataset_type]['out_dir'], suffix))
    for side in ['A', 'B']:
        idx_im_name = 0
        in_dir_size = size_dir_content(configs['Datasets'][dataset_type]['in_dir_'+side])
        transform = parse_transform(configs['Datasets'][dataset_type]['transforms_'+side])
        for im_name in tqdm(dir_content(configs['Datasets'][dataset_type]['in_dir_'+side], random=False)):
            im_raw = cv2.imread(os.path.join(configs['Datasets'][dataset_type]['in_dir_'+side], im_name))
            im_raw_transformed = transform(im_raw)
            test_or_train = np.random.random()
            if dataset_type == "SR_XR_complete":
                if im_raw.shape[0] < 700:
                    test_or_train = TEST
            elif dataset_type in ["XR_complete_2_XR_complete", "DRR_complete_2_XR_complete"]:
                pass

            if test_or_train < 0.9 * in_dir_size:
                suffix = "train"
            else:
                suffix = "test"
            cv2.imwrite(os.path.join(configs['Datasets'][dataset_type]['out_dir'], suffix+side, im_name), im_raw_transformed)
            idx_im_name+=1

if __name__ == '__main__':
    configs = SubXRParser()
    dataset_type = "DRR_complete_2_XR_complete"
    create_datasets(configs, dataset_type)
    #
    # data_path  = r'C:\Users\micha\PycharmProjects\CT_DRR\Data'
    # datasets_path  = r'C:\Users\micha\PycharmProjects\CT_DRR\Datasets'
    # # remove_and_create(os.path.join(datasets_path))
    # dir_content = lambda x,y: sorted(os.listdir(os.path.join(data_path, x,y)))
    #
    # input_dir =  dir_content('DRR','Input')
    # ulna_dir =   dir_content('DRR','Ulna')
    # radius_dir = dir_content('DRR','Radius')
    # xr_dir     = dir_content('X-Ray','')
    #
    # print (len(input_dir),len(ulna_dir),len(radius_dir))
    # assert (len(input_dir)==len(ulna_dir)==len(radius_dir))
    # n1 = len(input_dir)
    # n2 = len(xr_dir)
    # permutated_input_indexes = np.random.permutation(n1)
    # permutated_xr_indexes = np.random.permutation(n2)
    #
    # # train sets
    # for j in prange(12):
    #     print('DRR train set, augment ',j,'/12: ')
    #     for i in tqdm(permutated_input_indexes[:int(n1*0.9)]):
    #         f_name = input_dir[i]
    #         create_datasets(data_path,datasets_path,f_name,j,train=True)
    #
    #     # test sets
    #     print('DRR test set, augment ',j,'/12: ')
    #     for i in tqdm(permutated_input_indexes[int(n1*0.9):]):
    #         f_name = input_dir[i]
    #         create_datasets(data_path,datasets_path,f_name,j, train=False)
    #         # xr2ulna
    #
