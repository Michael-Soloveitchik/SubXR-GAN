import sys
import os
import shutil

import numpy as np
import torch
import os
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import cv2

from Models.Pix2Pix_single.models.networks import define_G
remove_and_create = lambda x: (not shutil.rmtree(x, ignore_errors=True)) and os.makedirs(x)
create_if_not_exists = lambda x: os.path.exists(x) or os.makedirs(x)

# XR_images_path  = r'C:\Users\micha\PycharmProjects\CT_DRR\Datasets\xr_real'
Models = {
    'drr2xr':[r'C:\Users\micha\PycharmProjects\CT_DRR\Models\pytorch-CycleGAN-and-pix2pix',r'runs\drr2xr_cyclegan1',r'C:\Users\micha\PycharmProjects\CT_DRR\Datasets\drr2xr\testA'],
    'xr2radius':[r'C:\Users\micha\PycharmProjects\CT_DRR\Models\Pix2Pix_single',r'runs\xr2radius',r'C:\Users\micha\PycharmProjects\CT_DRR\Datasets\xr2radius']

}
    # cmd1 = \
    #python ./Models/Pix2Pix-single/train.py --continue_train --epoch_count 6 --load_iter 2276 --checkpoints_dir ./Models/ --gan_mode lsgan --dataroot ./Datasets/xr2radius --name xr2radius --models pix2pix --batch_size 1 --save_by_iter --save_latest_freq 100 --save_epoch_freq 1 --continue_train --dataset_mode unaligned --direction AtoB --crop_size 512 --load_size 512 --no_flip --serial_batches --preprocess None
    #python ./Models/Pix2Pix-single/train.py --checkpoints_dir ./Models/ --gan_mode lsgan --dataroot ./Datasets/xr2radius --name xr2radius --models pix2pix --batch_size 1 --save_by_iter --save_latest_freq 100 --save_epoch_freq 1 --dataset_mode unaligned --direction AtoB --crop_size 512 --load_size 512 --no_flip --serial_batches --preprocess None
def get_model_and_dataset(model_name, epoch=None,test_set_path=None):
    model_pathes = Models[model_name]
    model_pathes[2]=test_set_path
    with open(os.path.join(model_pathes[0], 'options', 'train_options.py'), 'r') as f:
        train_keys = f.readlines()
        train_keys = [s.split('add_argument(')[1][3:] for s in train_keys if 'add_argument' in s]
        train_keys_indexes = [s.index("'") for s in train_keys]
        train_keys = [s[:i] for i,s in zip(train_keys_indexes,train_keys)] + \
                     ['dataroot','max_dataset_size','dataset_mode','checkpoints_dir']
    with open(os.path.join(model_pathes[0], 'options', 'test_options.py'), 'r') as f:
        test_keys = f.readlines()
        test_keys = [s.split('add_argument(')[1][3:] for s in test_keys if 'add_argument' in s]
        test_keys_indexes = [s.index("'") for s in test_keys]
        test_keys = [s[:i] for i, s in zip(test_keys_indexes, test_keys)]

        train_opt_path = os.path.join(*model_pathes[:2],'train_opt.txt')#r"C:\Users\micha\PycharmProjects\CT_DRR\Models\xr2radius\train_opt.txt"
        with open(train_opt_path, 'r') as f:
            options = f.readlines()[1:-1]
            options = [l.strip().split(':') for l in options]
            options = [[l[0],l[1][1:].split(' ')[0]] for l in options]
            options = ''.join([' --'+a[0]+' '+a[1] for a in options if (not (a[0] in train_keys) and (a[1] not in ['True','False']))])
            sys.argv = ('test.py' +''.join(options) +\
                        ' --dataroot ' + model_pathes[2] +\
                        ' --dataset_mode ' +'single' +\
                        ' --checkpoints_dir ' + os.path.join(model_pathes[0], 'runs')).split(' ')
        opt = TrainOptions().parse()
        opt.num_threads = 0  # test code only supports num_threads = 0
        opt.batch_size = 1  # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        opt.isTrain = False# get test options
        opt.epoch = str(epoch) if (epoch is not None) else 'latest'
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)
        model.setup(opt)
        if 'netG_A' in model.__dict__:
            G = model.netG_A
        else:
            G = model.netG
        return G, dataset
        # for i, data in enumerate(dataset):
            #     if i >= len(dataset):  # only apply our models to opt.num_test images.
            #         break
            #     visual = G(data['A'])
            #     # models.test()  # run inference
            #     # visuals = models.get_current_visuals()
            #     #
            #     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # # G.to(device)
            # images = torch.tensor(np.concatenate([torch.tensor((np.transpose(cv2.imread(os.path.join(XR_images_path, f)), [2, 0, 1]).astype(np.float32)[
            #                             np.newaxis, ...] / 256) - 0.5) for f in os.listdir(XR_images_path)],0))
            # processed_images = G(images)
