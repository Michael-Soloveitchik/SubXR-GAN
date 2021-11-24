import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.hub
# from Augmentations.augmentations import Augment_DRR
from numba import prange
import imgaug
import matplotlib.pyplot as plt
from Models.sr_gan.model import Generator
import cv2



def parse_transform(transform):
    transform = list(transform[0].items())
    if transform:
        transform_k, transfor_args = transform[0]
        if transform_k in TRANSFORMS:
            return lambda x :TRANSFORMS[transform_k](x, *transfor_args)
    return lambda x: x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
LR = 400
HR = 800
TR = 700
TEST = 1.1
SR_GAN = Generator().to(device)

def padding_transform(im, const=70):
    return np.pad(im,[(const,),(const,),(0,)],mode='constant')

def self_crop_transform(im):
    def get_circles_of_image(im):
        # gray-scalling
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        output = im + 0
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1000, param1=15, param2=10, minRadius=30, maxRadius=70)
        center = im.shape
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                center = (x,y)
                break
            # show the output image
            plt.imshow(output)
            plt.show()#np.hstack([im, output]))
        return center#np.median(Pangles)
    center = get_circles_of_image(im)
    h=min(min(center[1],480)/1.5,min(im.shape[0]-center[1],320))*2.5
    w=min(center[0],800)
    LR = int(min(h,w))
    im = im[center[1]-(LR-int(LR/2.22)):center[1]+int(LR/2.22),center[0]-LR:center[0]]
    return im, center

def self_rotate_transform(im):
    def get_orientation_of_image(im):
        # gray-scalling
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        # bluring
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

        # Canny edges
        low_threshold = 10
        high_threshold =100
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        # compute orientation
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 10  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 190  # minimum number of pixels making up a line
        max_line_gap = 40  # maximum gap in pixels between connectable line segments

        line_image = np.copy(im) * 0  # creating a blank to draw lines on
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        Plines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                 min_line_length, max_line_gap)
        Pangles = []
        Pdists = []
        for line in Plines:
            for x1,y1,x2,y2 in line:
                if x1>x2:
                    tx,ty=x1+0,y1+0
                    x1,y1=x2+0,y2+0
                    x2,y2=tx+0,ty+0
                if (x2-x1)>1e-05:
                    Pangles.append(np.arctan((y2-y1)/(x2-x1))/np.pi*180)
                else:
                    Pangles.append(90)
                # Pdists.append(np.sqrt((x1-x2)**2+(y1-y2)**2))
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            # Draw the lines on the  image
        lines_edges = cv2.addWeighted(im, 0.8, line_image, 1, 0)
        # plt.imshow(lines_edges)
        # plt.show()
        # plt.hist(Pangles)
        # plt.show()
        return np.median(Pangles)

    angle, new_angle = 0,1
    cum_angle=0
    while np.abs(angle-new_angle)> 1e-01:
        angle = get_orientation_of_image(im)
        im = rotate_transform(im, angle)
        cum_angle+=angle
        new_angle = get_orientation_of_image(im)
    return im, cum_angle

def flip_rotate_90_counter_clockwisw(im):
    im=cv2.transpose(im)
    im=cv2.flip(im,flipCode=1)
    return im
def rotate_transform(nh_i_im, angle):
    nh_i_im_g = cv2.cvtColor(nh_i_im, cv2.COLOR_BGR2GRAY) if len(nh_i_im.shape) > 2 else nh_i_im
    if np.abs(np.abs(angle)-90) < 45:
        nh_i_im_g = flip_rotate_90_counter_clockwisw(nh_i_im_g)
        angle = np.sign(angle)*np.abs((np.abs(angle)-90))
    # Detect keypoints (features) cand calculate the descriptors
    cy, cx = nh_i_im_g.shape
    center = (cx // 2, cy // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    h_i_im_g = cv2.warpAffine(src=nh_i_im_g, M=M, dsize=(nh_i_im_g.shape[1], nh_i_im_g.shape[0]))
    h_i_im_g_bgr = cv2.cvtColor(h_i_im_g, cv2.COLOR_GRAY2BGR)
    return h_i_im_g_bgr
def crop_transform(im, H=800,W=800):
    im_c_h, im_c_w = im.shape[0]//2, im.shape[1]//2
    im = im[im_c_h-H//2:im_c_h+H//2, im_c_w-W//2:im_c_w+W//2]
    return im
def translate_transform(im, TR=-15):
    im_new = im * 0.
    im_new[max(0,TR):im_new.shape[0]+TR,max(0,TR):im_new.shape[1]+TR]=im[max(0,TR):im_new.shape[0]+TR,max(0,TR):im_new.shape[1]+TR]
    return im_new
def resize_transform(im, H,W):
    LR_im = cv2.resize(im, (H, W))
    return LR_im

def numpy_2_torch_transform(im):
    torch_im = (torch.from_numpy(np.transpose(im, [2, 0, 1])[None]) / 255.).to(device)

    return torch_im

def torch_2_numpy_transform(im):
    np_im = np.transpose(im[0].float().cpu().detach().numpy(), [1, 2, 0])*255.
    return np_im

def super_resolution_transform(im, SR_GAN_path, LR=384, HR=768, TR=650):
    SR_GAN.load_state_dict(torch.load(SR_GAN_path, map_location=device))
    SR_GAN.eval()
    # SR_GAN.half()
    if im.shape[1]<TR:
        LR_im = resize_transform(im, LR,LR)
        torch_LR_im = numpy_2_torch_transform(LR_im)
        torch_HR_im = SR_GAN(torch_LR_im)
        HR_im = torch_2_numpy_transform(torch_HR_im)
    else:
        HR_im = resize_transform(im, HR,HR)
    return HR_im

def drr_2_xr_style_transform(im, G_style_path, LR=400, HR=800, TR=700):
    SR_GAN.load_state_dict(torch.load(G_style_path, map_location=device))
    SR_GAN.eval()
    # SR_GAN.half()
    if im.shape[1]<TR:
        LR_im = resize_transform(im, LR,LR)
        torch_LR_im = numpy_2_torch_transform(LR_im)
        torch_HR_im = SR_GAN(torch_LR_im)
        HR_im = torch_2_numpy_transform(torch_HR_im)
    else:
        HR_im = resize_transform(im, HR,HR)
    return HR_im


TRANSFORMS = {
    "down_sample":      resize_transform,
    "crop":             crop_transform,
    "translate":        translate_transform,
    "up_sample":        resize_transform,
    "SR_GAN":           super_resolution_transform,
    "subtraction_AU":   super_resolution_transform,
}
