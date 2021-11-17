import sys
import os
import shutil
import cv2
from utils import remove_and_create, create_if_not_exists
from SubXR_configs_parser import SubXRParser

def initialze(configs):
    header = "Data"
    for data_type in configs[header]:
        for data_source in configs[header][data_type]:
            if configs[header][data_type][data_source]['in_dir']:
                remove_and_create(configs[header][data_type][data_source]['out_dir'])
                if data_source == "DRR":
                    for dir in os.listdir(configs[header][data_type][data_source]['in_dir']):
                        if not os.path.isdir(dir):
                            continue
                        full_in_drr_path = os.path.join(configs[header][data_type][data_source]['in_dir'], dir)
                        for dir in configs[header][data_type][data_source]['in_sub_folders']:
                            if not os.path.isdir(dir):
                                continue
                            create_if_not_exists(os.path.join(full_in_drr_path, 'pre_DRR',dir))
                    for dir in configs[header][data_type][data_source]['out_sub_folders']:
                        if not os.path.isdir(dir):
                            continue
                        create_if_not_exists(os.path.join(configs[header][data_type][data_source]['out_dir'],dir))
    header = "Datasets"
    for dataset_type in configs[header]:
        for sub_folder in configs[header][dataset_type]['out_sub_folders']:
            remove_and_create(os.path.join(configs[header][dataset_type]['out_dir'],sub_folder))

if __name__ == '__main__' :
    configs = SubXRParser()
    initialze(configs)
