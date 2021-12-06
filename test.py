import cv2
import numpy as np
from matplotlib import pyplot as plt
from Transforms.transforms import *
if __name__=='__main__':
    img = cv2.imread(r"C:\Users\micha\Research\SubXR-GAN\Data\DRR_complete\Ulna_mask\Ulna_mask_00022.jpg")
    img3=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(plt.hist(img3.flatten()))
    plt.figure(3)
    plt.imshow(img3)

    img4 = cv2.threshold(img3, 229, 255,cv2.THRESH_BINARY)[1]
    print(plt.hist(img4.flatten()))
    plt.figure(4)
    plt.imshow(img4)
    # img,angle = self_rotate_transform(img)
    # img,center = self_crop_transform(img)
    # img = padding_transform(img)
    plt.imshow(img)
    plt.show()
if 0 and  __name__=='__main__':
    img = cv2.imread(r"G:\My Drive\CT-DRR\Data\raw X-Ray - Data\x rays drf\SynapseExport (5)\Image00002.jpg")
    img,angle = self_rotate_transform(img)
    img,center = self_crop_transform(img)
    img = padding_transform(img)
    plt.imshow(img)
    plt.show()
