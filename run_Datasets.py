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
from Transforms.transforms import *


CROP_SIZE = 768




def create_datasets(configs, dataset_type):
    remove_and_create(os.path.join(configs['Datasets'][dataset_type]['out_dir']))
    # create_if_not_exists(os.path.join(configs['Datasets'][dataset_type]['out_dir']))
    for suffix in configs['Datasets'][dataset_type]['out_sub_folders']:
        remove_and_create(os.path.join(configs['Datasets'][dataset_type]['out_dir'], suffix))
        create_if_not_exists(os.path.join(configs['Datasets'][dataset_type]['out_dir'], suffix))
    K = 20
    in_dir_A_size = size_dir_content(configs['Datasets'][dataset_type]['in_dir_A'])
    in_dir_B_size = size_dir_content(configs['Datasets'][dataset_type]['in_dir_B'])
    seeds_permutations = np.random.permutation(max(in_dir_A_size,in_dir_B_size))*K
    for side in ['A', 'B']:
        idx_im_name = 0
        augmentation = parse_augmentation(configs['Datasets'][dataset_type]['augmentation_'+side])
        in_dir_size = size_dir_content(configs['Datasets'][dataset_type]['in_dir_'+side])
        transform = parse_transforms(configs['Datasets'][dataset_type]['transform_' + side],dataset_type=dataset_type)
        for i, im_name in enumerate(tqdm(dir_content(configs['Datasets'][dataset_type]['in_dir_'+side], random=False))):
            im_raw = cv2.imread(os.path.join(configs['Datasets'][dataset_type]['in_dir_'+side], im_name))
            seeds = np.arange(seeds_permutations[i], seeds_permutations[i] + K)
            im_raw_transformed = transform(im_raw,im_name=im_name)
            for seed in seeds:
                random.seed(seed);
                np.random.seed(seed);
                imgaug.random.seed(seed)
                test_or_train = np.random.random()
                im_raw_transformed_augmented = augmentation(im_raw_transformed)
                if dataset_type == "SR_XR_complete":
                    if im_raw.shape[0] < 700:
                        test_or_train = TEST
                elif dataset_type in ["XR_complete_2_XR_complete", "DRR_complete_2_XR_complete"]:
                    pass

                out_dir_size = size_dir_content(os.path.join(configs['Datasets'][dataset_type]['out_dir'], "train"+side))
                if test_or_train < 0.9 or ((test_or_train<1.0) and (0.9*in_dir_size*10 <= out_dir_size)) :
                    suffix = "train"
                else:
                    suffix = "test"
                im_transformed_augmented_name = im_name.split('_')[0]+'_{idx_im_name}.jpg'.format(idx_im_name=idx_im_name)
                cv2.imwrite(os.path.join(configs['Datasets'][dataset_type]['out_dir'], suffix+side, im_transformed_augmented_name), im_raw_transformed_augmented)
                idx_im_name+=1

if __name__ == '__main__':
    configs = SubXRParser()
    # create_datasets(configs, "SR_XR_complete")
    # create_datasets(configs, "DRR_complete_2_XR_complete")
    create_datasets(configs, "XR_complete_2_Radius_mask")
    # create_datasets(configs, "XR_complete_2_Ulna_mask")
    #
