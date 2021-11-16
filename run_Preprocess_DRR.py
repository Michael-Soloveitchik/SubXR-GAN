import sys
import os
import shutil

import Preprocessing.crop
from Preprocessing.rotate import regularize_orientation
from Models.models_inference import get_model_and_dataset
import cv2
from util import util
from tqdm import tqdm
from Preprocessing.crop import crop
if __name__ == '__main__':
    DATA_apth  = ' '.join(sys.argv[1:])
    for s in ['Input', 'Ulna', 'Radius']:
        totall = 0
        # print(totall, i * n)

        for dir_path in os.listdir(DATA_apth):
            dataset_path = os.path.join(r'C:\Users\micha\PycharmProjects\CT_DRR\Data', 'DRR')
            DRR_path = os.path.join(DATA_apth, dir_path, 'DRR')
            pre_DRR_path = os.path.join(DATA_apth, dir_path, 'pre_DRR')

            in_dir_totall = 0

            for i, suf in enumerate(['X', 'Y', 'XY', 'YX']):
                n = len(
                    [t for t in os.listdir(os.path.join(pre_DRR_path, suf)) if t.endswith('.png') and t.startswith(s)])
                j = 0
                for f in tqdm(sorted(os.listdir(os.path.join(pre_DRR_path, suf)))):

                    if f.endswith('.png') and f.startswith(s):
                        new_totall = ("{0:5d}".format(totall + i * n + j) + '.jpg').replace(' ', '0')
                        shutil.copy(os.path.join(pre_DRR_path, suf, f), os.path.join(dataset_path, s, new_totall))
                        if suf in ['XY', 'YX']:
                            regularize_orientation([os.path.join(dataset_path, s, new_totall)])
                        crop(os.path.join(dataset_path, s, new_totall))
                        j += 1
                        in_dir_totall += 1
            totall += in_dir_totall

        G, dataset = get_model_and_dataset('drr2xr', 316, os.path.join(dataset_path, s))
        for data in tqdm(dataset):
            if data['A_paths'][0].endswith('01350.jpg'):
                pass
            cycled_im = G(data['A'])
            im = util.tensor2im(cycled_im)
            cv2.imwrite(data['A_paths'][0], im)
