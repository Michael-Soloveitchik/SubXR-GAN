import cv2
import numpy as np
from matplotlib import pyplot as plt
from Transforms.transforms import *

if __name__=='__main__':
    img = cv2.imread(r"G:\My Drive\CT-DRR\Data\raw X-Ray - Data\x rays drf\SynapseExport (5)\Image00002.jpg")
    img,angle = self_rotate_transform(img)
    img,center = self_crop_transform(img)
    img = padding_transform(img)
    plt.imshow(img)
    plt.show()
