1
from Augmentations.augmentations import Augment_DRR
import sys
import os

if __name__ == '__main__':
    DATA_apth  = ' '.join(sys.argv[1:])
    for dir_path in os.listdir(DATA_apth):
        for suf in ['Input','Ulna','Radius']:
            full_drr_path = os.path.join(DATA_apth, dir_path,'DRR',suf)
            Augment_DRR(full_drr_path)