import numpy as np
import os
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import random
import imgaug
win_size =512
def Augment_DRR(drr_path, file_name, j,seed_n):

    im = cv2.imread(os.path.join(drr_path,file_name))
    random.seed(seed_n)
    np.random.seed(seed_n)
    imgaug.random.seed(seed_n)
    for AU in [Au1]:

        augmented_im = AU(image=im)['image']
        plt.imshow(augmented_im)
        new_filename = ("{0:2d}".format(j) + file_name).replace(' ', '0')
        desired_size = 512
        im = augmented_im #cv2.imread(im_pth)
        old_size = im.shape[:2]  # old_size is in (height, width) format
        ratio = float(desired_size) / max(old_size)
        # new_size = tuple([int(x * ratio) for x in old_size])
        # im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_augmented_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        im = cv2.imwrite(os.path.join(drr_path, new_filename),new_augmented_im)
            # plt.show()



Au1 = A.Compose([

        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.2, rotate_limit=15, p=0.6),
        A.OneOf([
            # A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.Perspective (p=0.3),
        ], p=0.4),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.4),
        A.HueSaturationValue(p=0.3),
    ])