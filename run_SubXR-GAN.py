import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from Transforms.transforms import *
from run_GIFs import create_GIF
import shutil

if 0 and  __name__=='__main__':


    path = sys.argv
    if os.path.isdir(path):
        os.makedirs(os.path.join(path))
    else:
        image_path = path

    img = cv2.imread(img_path)
    # Fix image orientation
    img,angle = self_rotate_transform(img)
    img,center = self_crop_transform(img)
    img = padding_transform(img)


    # Cycle-GAN activation
    shape_inp = im.shape[0]
    style_transform_name = model_dir.split('\\')[-1]
    model_dir = '\\'.join(model_dir.split('\\')[:-2])
    cmd = (f"python {model_dir}\\test.py --model_suffix _A --dataroot {temp_drr2xr_dir} --name . --load_iter {iter} --netG unet_128 --checkpoints_dir {os.path.join(model_dir, 'SAMPLES',style_transform_name)} --model test --no_dropout --crop_size {shape_inp} --results_dir {temp_drr2xr_dir}")
    subprocess.call(cmd)
    xr_im = cv2.imread(os.path.join(temp_drr2xr_dir,f'test_latest_iter{iter}','images','drr2xr_fake.png'))
    shutil.rmtree(temp_drr2xr_dir)



    create_GIFs(path)
    plt.imshow(img)
    plt.show()
