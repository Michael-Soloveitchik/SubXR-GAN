import cv2
import numpy as np
from matplotlib import pyplot as plt
from Transforms.transforms import *
if 0 and __name__=='__main__':
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
    plt.figure(0)
    plt.imshow(img)
    img,angle = self_rotate_transform(img)
    plt.figure(1)
    plt.imshow(img)
    img,center = self_crop_transform(img)
    plt.figure(2)
    plt.imshow(img)
    img = padding_transform(img)
    plt.imshow(img)
    plt.show()
if 1 and __name__ == '__main__':
    img1 = cv2.imread(r"C:\Users\micha\Research\SubXR-GAN\Datasets\drr_complete_2_xr_complete\trainA\Input_52.jpg")
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(r"C:\Users\micha\Research\SubXR-GAN\Datasets\drr_complete_2_xr_complete\trainA\Input_41.jpg")
    img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3 = cv2.imread(r"C:\Users\micha\Research\SubXR-GAN\Datasets\drr_complete_2_xr_complete\trainB\xr_4.jpg")
    img3=cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img4 = cv2.imread(r"C:\Users\micha\Research\SubXR-GAN\Datasets\drr_complete_2_xr_complete\trainB\xr_774.jpg")
    img4=cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    for i, im in enumerate([img1,img2,img3,img4]):
        plt.figure(i)
        plt.hist(im.flatten())
        print(im.flatten().mean())
        print(im.flatten().var())
        print('___')
    plt.show()
    print('Shalom')