from Transforms.transforms import *
import os
import cv2
if __name__=='__main__':
    in_dir  = r"G:\My Drive\CT-DRR\Data\raw X-Ray - Data\x rays drf"
    out_dir = r"G:\My Drive\CT-DRR\Data\clear X-Ray - Data\x rays drf"
    os.path.exists(out_dir) or os.makedirs(out_dir)
    for xr_dir in os.listdir(in_dir):
        if not os.path.isdir(os.path.join(out_dir, xr_dir)):
            continue
        os.path.exists(os.path.join(out_dir, xr_dir)) or os.makedirs(os.path.join(out_dir, xr_dir))
        for xr_im in os.listdir(os.path.join(in_dir, xr_dir)):
            if not xr_im.endswith('.jpg'):
                continue
            img = cv2.imread(os.path.join(in_dir, xr_dir,xr_im))
            # plt.figure(0)
            # plt.imshow(img)
            img,angle = self_rotate_transform(img)
            # plt.figure(1)
            # plt.imshow(img)
            img,center = self_crop_transform(img)
            # plt.figure(2)
            # plt.imshow(img)
            # img = padding_transform(img)
            cv2.imwrite(os.path.join(out_dir, xr_dir,xr_im), img)
            plt.figure(3)
            plt.imshow(img)
            plt.show()