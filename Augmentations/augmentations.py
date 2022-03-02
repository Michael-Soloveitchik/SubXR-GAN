import numpy as np
import os
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import random
import imgaug

sr_xr_complete_AU = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1,rotate_limit=10, p=0.5),
    A.Sharpen(),
    # A.Emboss(),
    # A.RandomBrightnessContrast(),
    # A.Perspective (p=0.3),
    ])
drr_complete_2_xr_complete_AU = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.5, rotate_limit=20, p=1.0),
    # A.Perspective (p=0.3),
    A.Sharpen(),
    A.VerticalFlip(0.5),
    A.RandomCrop(64, 64, always_apply=True, p=1.0)
    # A.Emboss(),
    # A.RandomBrightnessContrast(),
])
full_xr_AU = sr_xr_complete_AU


A.Compose([

    A.OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.3),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.2, rotate_limit=10, p=0.6),
    A.OneOf([
        # A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=.1),
        A.Perspective (p=0.3),
    ], p=0.4),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen(),
        # A.Emboss(),
        # A.RandomBrightnessContrast(),
    ], p=0.4),
    A.HueSaturationValue(p=0.3),
])

OLD = A.Compose([

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
        A.Sharpen(),
        # A.Emboss(),
        # A.RandomBrightnessContrast(),
    ], p=0.4),
    A.HueSaturationValue(p=0.3),
])


AUGMENTATIONS = {
    "sr_xr_complete_AU"             : sr_xr_complete_AU,
    "drr_complete_2_xr_complete_AU" : drr_complete_2_xr_complete_AU,
    "full_xr_AU"                    : full_xr_AU
}

def parse_augmentation(augmentation):
    augmentation_k = augmentation
    if augmentation:
        if augmentation_k in AUGMENTATIONS:
            return lambda x: AUGMENTATIONS[augmentation_k](image=x)['image']
    return lambda x: x
