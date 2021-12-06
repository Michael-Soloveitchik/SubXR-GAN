import os.path

from utils import *
from SubXR_configs_parser import SubXRParser
from tqdm import tqdm
from Transforms.transforms import *
import cv2


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

                    transform = parse_transforms(configs['Data'][data_type][data_source]['transform'],data_type)

                    for dir in tqdm(dir_content(configs['Data'][data_type][data_source]['in_dir'],random=False)):
                        if not os.path.isdir(os.path.join(configs['Data'][data_type][data_source]['in_dir'],dir)):
                            continue
                        for prefix in configs["Data"][data_type][data_source]['out_sub_folders']:
                            i = 0
                            for orientation in configs["Data"][data_type][data_source]['in_sub_folders']:
                                pre_DRR_orientation_path = os.path.join(configs['Data'][data_type][data_source]['in_dir'], dir,
                                                            "pre_DRR",orientation)
                                if 'mask' not in prefix:
                                    prefix_files = sorted([f for f in os.listdir(pre_DRR_orientation_path) if f.startswith(prefix) and f.endswith('.png') and (not 'mask' in f)])
                                else:
                                    prefix_files = sorted([f for f in os.listdir(pre_DRR_orientation_path) if f.startswith(prefix) and f.endswith('.png') and ('mask' in f)])
                                for im_name in tqdm(prefix_files):
                                    new_im_name = prefix+'_'+str(i).zfill(5)+".jpg"
                                    shutil.copy(os.path.join(pre_DRR_orientation_path, im_name),
                                                os.path.join(configs['Data'][data_type][data_source]['out_dir'], prefix, new_im_name))
                                    rotated_im_path = os.path.join(configs['Data'][data_type][data_source]['out_dir'], prefix, new_im_name)
                                    nh_i_im = cv2.imread(rotated_im_path)
                                    if ('XY' in orientation):
                                        nh_i_im = rotate_transform(nh_i_im, -20,data_type,im_name)
                                    if ('YX' in orientation):
                                        nh_i_im = rotate_transform(nh_i_im, 20,data_type,im_name)
                                    nh_i_im = transform(nh_i_im,im_name)
                                    cv2.imwrite(rotated_im_path, nh_i_im)
                                    i+=1
                                    # crop(os.path.join(dataset_path, s, new_totall))

if __name__ == '__main__':
    configs = SubXRParser()
    import_data(configs)
