import os.path

from utils import *
from SubXR_configs_parser import SubXRParser
from tqdm import tqdm
import cv2

def regularize_orientation(full_nh_i_im_path):
    nh_i_im = cv2.imread(full_nh_i_im_path)
    nh_i_im_g = cv2.cvtColor(nh_i_im, cv2.COLOR_BGR2GRAY) if len(nh_i_im.shape) > 2 else nh_i_im

    # Detect keypoints (features) cand calculate the descriptors
    cy, cx = nh_i_im_g.shape
    center = (cx // 2, cy // 2)
    M = cv2.getRotationMatrix2D(center, -38, 1)
    h_i_im_g = cv2.warpAffine(src=nh_i_im_g, M=M, dsize=(nh_i_im_g.shape[1], nh_i_im_g.shape[0]))
    cv2.imwrite(full_nh_i_im_path, h_i_im_g)

CROP = 30
def import_data(configs):
    for data_type in configs['Data']:
        for data_source in configs['Data'][data_type]:
            if configs['Data'][data_type][data_source]['in_dir']:
                if data_source == 'XR':
                    i = 0
                    for dir in tqdm(dir_content(configs['Data'][data_type][data_source]['in_dir'], random=False)):
                        if not os.path.isdir(os.path.join(configs['Data'][data_type][data_source]['in_dir'],dir)):
                            continue
                        for im_path in os.listdir(os.path.join(configs['Data'][data_type][data_source]['in_dir'],dir)):
                            im_in = cv2.imread(os.path.join(configs['Data'][data_type][data_source]['in_dir'],dir,im_path))
                            if im_in is None:
                                continue
                            im_out = im_in[CROP:-CROP,CROP:-CROP]
                            new_im_path = "xr_"+str(i).zfill(4)+".jpg"
                            cv2.imwrite(os.path.join(configs['Data'][data_type][data_source]['out_dir'],new_im_path),im_out )
                            # shutil.copy(os.path.join(configs['Data'][data_type][data_source]['in_dir'],dir,im_path), os.path.join(configs['Data'][data_type][data_source]['out_dir'],new_im_path))
                            i += 1
                elif data_source == 'DRR':
                    i = 0
                    for dir in tqdm(dir_content(configs['Data'][data_type][data_source]['in_dir'],random=False)):
                        if not os.path.isdir(os.path.join(configs['Data'][data_type][data_source]['in_dir'],dir)):
                            continue
                        for prefix in configs["Data"][data_type][data_source]['out_sub_folders']:
                            for orientation in configs["Data"][data_type][data_source]['in_sub_folders']:
                                pre_DRR_orientation_path = os.path.join(configs['Data'][data_type][data_source]['in_dir'], dir,
                                                            "pre_DRR",orientation)
                                prefix_files = sorted([f for f in os.listdir(pre_DRR_orientation_path) if f.startswith(prefix) and f.endswith('.png')])
                                for im_name in tqdm(prefix_files):
                                    new_im_name = str(i).zfill(5)+".jpg"
                                    shutil.copy(os.path.join(pre_DRR_orientation_path, im_name),
                                                os.path.join(configs['Data'][data_type][data_source]['out_dir'], prefix, new_im_name))
                                    if orientation in ['XY', 'YX']:
                                        regularize_orientation(os.path.join(configs['Data'][data_type][data_source]['out_dir'], prefix, new_im_name))

                                    i+=1
                                    # crop(os.path.join(dataset_path, s, new_totall))

if __name__ == '__main__':
    configs = SubXRParser()
    import_data(configs)
