import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

def regularize_orientation(full_nh_i_im_paths_arr):
        full_nh_i_im_path = full_nh_i_im_paths_arr[0]
        nh_i_im = cv2.imread(full_nh_i_im_path)
        nh_i_im_g = cv2.cvtColor(nh_i_im, cv2.COLOR_BGR2GRAY) if len(nh_i_im.shape)>2 else nh_i_im

        # Detect keypoints (features) cand calculate the descriptors
        cy,cx = nh_i_im_g.shape
        center = (cx//2,cy//2)
        M = cv2.getRotationMatrix2D(center,-38,1)
        h_i_im_g = cv2.warpAffine(src=nh_i_im_g, M=M, dsize=(nh_i_im_g.shape[1], nh_i_im_g.shape[0]))
        for output in full_nh_i_im_paths_arr:
            cv2.imwrite(output,h_i_im_g)


