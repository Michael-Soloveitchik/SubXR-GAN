import sys
import os
import shutil

from tqdm import tqdm
from SubXR_configs_parser import SubXRParser
import cv2
from utils import *
import numpy as np
import utils
from Augmentations.augmentations import *
from Transfers.transfers import *



TRANSFORMS = {
    "down_sample": downsample_transform,
    "up_sample":   upsample_transform,
    "SR_GAN":      super_resolution_transform,
}

AUGMENTATIONS = {
    "sr_xr_complete_AU":        sr_xr_complete_AU,
    "sr_xr_complete_AU":        sr_xr_complete_AU,
    "sr_xr_complete_AU":        sr_xr_complete_AU
}
def parse_transform(transform):
    transform = list(transform.items())
    if transform:
        transform_k, transfor_args = transform[0]
        if transform_k in TRANSFORMS:
            return lambda x :TRANSFORMS[transform_k](x, *transfor_args)
    return lambda x: x

def parse_augmentation(augmentation):
    augmentation_k = augmentation
    if augmentation:
        if augmentation_k in AUGMENTATIONS:
                return lambda x,seed : random.seed(seed); np.random.seed(seed); imgaug.random.seed(seed); AUGMENTATIONS[augmentation_k](image=x)['image']
    return lambda x, seed: x

def create_datasets(configs, dataset_type):
    for suffix in configs['Datasets'][dataset_type]['out_sub_folders']:
        remove_and_create(os.path.join(configs['Datasets'][dataset_type]['out_dir'], suffix))
        create_if_not_exists(os.path.join(configs['Datasets'][dataset_type]['out_dir'], suffix))
    for side in ['A', 'B']:
        idx_im_name = 0
        transform = parse_transform(configs['Datasets'][dataset_type]['transform_'+side])
        augmentation = parse_augmentation(configs['Datasets'][dataset_type]['augmentation_'+side])
        in_dir_size = size_dir_content(configs['Datasets'][dataset_type]['in_dir_'+side])
        for im_name in tqdm(dir_content(configs['Datasets'][dataset_type]['in_dir_'+side], random=False)):
            for seed in range(10):
                im_raw = cv2.imread(os.path.join(configs['Datasets'][dataset_type]['in_dir_'+side], im_name))

                im_raw_transformed = transform(im_raw)

                im_raw_transformed_augmented = augmentation(im_raw_transformed,seed)

            test_or_train = np.random.random()
            if dataset_type == "SR_XR_complete":
                if im_raw.shape[0] < 700:
                    test_or_train = TEST
            elif dataset_type in ["XR_complete_2_XR_complete", "DRR_complete_2_XR_complete"]:
                pass

            out_dir_size = size_dir_content(os.path.join(configs['Datasets'][dataset_type]['out_dir'], "train"+side))
            if test_or_train < 0.9 or ((test_or_train<1.0) and (0.9*in_dir_size <= out_dir_size)) :
                suffix = "train"
            else:
                suffix = "test"
            cv2.imwrite(os.path.join(configs['Datasets'][dataset_type]['out_dir'], suffix+side, im_name), im_raw_transformed_augmented)
            idx_im_name+=1

if __name__ == '__main__':
    configs = SubXRParser()
    dataset_type = "DRR_complete_2_XR_complete"
    create_datasets(configs, "SR_XR_complete")
    create_datasets(configs, "DRR_complete_2_XR_complete")
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
