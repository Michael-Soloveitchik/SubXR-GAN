import sys
import os
import shutil
import cv2
from utils import remove_and_create, create_if_not_exists
from SubXR_configs_parser import SubXRParser
from utils import *
def initialze(configs):
    header = "Data"
    for data_type in configs["Data"]:
        for data_source in configs["Data"][data_type]:
            if configs["Data"][data_type][data_source]['in_dir']:
                remove_and_create(configs["Data"][data_type][data_source]['out_dir'])
                for patient_dir in dir_content(configs["Data"][data_type][data_source]['in_dir'], random=False):
                    if not os.path.isdir(os.path.join(configs["Data"][data_type][data_source]['in_dir'], patient_dir)):
                        continue
                    for dir in configs["Data"][data_type][data_source]['in_sub_folders']:
                        create_if_not_exists(os.path.join(configs["Data"][data_type][data_source]['in_dir'],patient_dir, 'pre_DRR',dir))
                    for dir in configs["Data"][data_type][data_source]['out_sub_folders']:
                        create_if_not_exists(os.path.join(configs["Data"][data_type][data_source]['out_dir'],dir))

    for dataset_type in configs["Datasets"]:
        for sub_folder in configs["Datasets"][dataset_type]['out_sub_folders']:
            remove_and_create(os.path.join(configs["Datasets"][dataset_type]['out_dir'], sub_folder))

if __name__ == '__main__' :
    configs = SubXRParser()
    initialze(configs)
