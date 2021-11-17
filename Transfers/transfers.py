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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
LR = 400
HR = 800
TR = 700
TEST = 1.1
SR_GAN = Generator().to(device)

def downsample_transform(im, LR=400):
    LR_im = cv2.resize(im, (LR, LR))
    return LR_im

def upsample_transform(im, HR=700):
    LR_im = cv2.resize(im, (HR, HR))
    return LR_im

def numpy_2_torch_transform(im):
    torch_im = (torch.from_numpy(np.transpose(im, [2, 0, 1])[None]) / 255.).to(device)

    return torch_im

def torch_2_numpy_transform(im):
    np_im = np.transpose(im[0].float().cpu().detach().numpy(), [1, 2, 0])*255.
    return np_im

def super_resolution_transform(im, SR_GAN_path, LR=400, HR=800, TR=700):
    SR_GAN.load_state_dict(torch.load(SR_GAN_path, map_location=device))
    SR_GAN.eval()
    # SR_GAN.half()
    if im.shape[1]<TR:
        LR_im = downsample_transform(im, LR)
        torch_LR_im = numpy_2_torch_transform(LR_im)
        torch_HR_im = SR_GAN(torch_LR_im)
        HR_im = torch_2_numpy_transform(torch_HR_im)
    else:
        HR_im = upsample_transform(im, HR)
    return HR_im