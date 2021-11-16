import sys
import os
import shutil
import numpy as np
import random
from Augmentations.augmentations import Augment_DRR
from numba import prange
import imgaug
from tqdm import tqdm
import cv2
remove_and_create = lambda x: (not shutil.rmtree(x, ignore_errors=True)) and os.makedirs(x)

if __name__ == '__main__':
    data_path  = r'C:\Users\micha\PycharmProjects\CT_DRR\Data'
    datasets_path  = r'C:\Users\micha\PycharmProjects\CT_DRR\Datasets'
    DRR_dir = os.path.join(datasets_path, 'xr2radius', 'trainA')
    # remove_and_create(os.path.join(datasets_path))
    dir_content = lambda x,y: sorted(os.listdir(os.path.join(data_path, x,y)))

    input_dir =  dir_content('DRR','Input')
    ulna_dir =   dir_content('DRR','Ulna')
    radius_dir = dir_content('DRR','Radius')
    xr_dir     = dir_content('X-Ray','')
    drr_dir     = input_dir #dir_content(DRR_dir,'')

    print (len(input_dir),len(ulna_dir),len(radius_dir))
    assert (len(input_dir)==len(ulna_dir)==len(radius_dir))
    n1 = len(drr_dir)
    n2 = len(xr_dir)
    permutated_input_indexes = np.random.permutation(n1)
    permutated_xr_indexes = np.random.permutation(n2)

    # DRR2XR
    remove_and_create(os.path.join(datasets_path, 'drr2xr'))
    remove_and_create(os.path.join(datasets_path, 'drr2xr', 'trainA'))
    remove_and_create(os.path.join(datasets_path, 'drr2xr', 'trainB'))
    remove_and_create(os.path.join(datasets_path, 'drr2xr', 'testA'))
    remove_and_create(os.path.join(datasets_path, 'drr2xr', 'testB'))

    # XR
    remove_and_create(os.path.join(datasets_path, 'xr_real'))
    print('X-Ray train set:')
    for i in tqdm(permutated_input_indexes[:int(n1 * 0.9)]):
        f_name = input_dir[i]
        shutil.copy(os.path.join(data_path,'DRR','Input', f_name),
                os.path.join(datasets_path, 'drr2xr', 'trainA'))
    for i in tqdm(permutated_xr_indexes[:int(n2*0.9)]):
        f_name = xr_dir[i]
        shutil.copy(os.path.join(data_path, 'X-Ray', f_name),
                os.path.join(datasets_path, 'drr2xr', 'trainB'))
    # test sets python train.py --dataroot C:\Users\micha\PycharmProjects\CT_DRR\Datasets\drr2xr --load_size 512 --crop_size 512 --display_winsize 512 --gan_mode lsgan --name drr2xr_cyclegan --models cycle_gan --batch_size 10 --verbose --num_threads 0
    print('X-Ray test set:')
    for i in tqdm(permutated_input_indexes[int(n1*0.9):]):
        f_name = input_dir[i]
        shutil.copy(os.path.join(data_path,'DRR','Input', f_name),
                os.path.join(datasets_path, 'drr2xr', 'testA'))
    for i in tqdm(permutated_xr_indexes[int(n2*0.9):]):
        f_name = xr_dir[i]
        shutil.copy(os.path.join(data_path, 'X-Ray', f_name),
                    os.path.join(datasets_path, 'drr2xr', 'testB'))

    for i in tqdm(permutated_xr_indexes):
        f_name = xr_dir[i]
        shutil.copy(os.path.join(data_path, 'X-Ray', f_name),
                    os.path.join(datasets_path, 'xr_real'))

        # xr2ulna




