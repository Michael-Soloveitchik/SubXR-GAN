import sys
import os
import shutil
import numpy as np
import random
from Augmentations.augmentations import Augment_DRR
from numba import prange
import imgaug
from tqdm import tqdm

# remove_and_create = lambda x: (not shutil.rmtree(x, ignore_errors=True)) and os.makedirs(x)

def create_datasets(data_path,datasets_path,f_name, j, train=True):
    seed_n = random.randint(0, 2 ** 32 - 1)

    # xr2ulna
    shutil.copy(os.path.join(data_path, 'DRR', 'Input', f_name),
                os.path.join(datasets_path, 'xr2ulna', 'trainA' if train else 'testA'))
    Augment_DRR(os.path.join(datasets_path, 'xr2ulna', 'trainA' if train else 'testA'), f_name,j,seed_n)
    os.remove(os.path.join(datasets_path, 'xr2ulna', 'trainA' if train else 'testA', f_name))

    shutil.copy(os.path.join(data_path, 'DRR', 'Ulna', f_name),
                os.path.join(datasets_path, 'xr2ulna', 'trainB' if train else 'testB'))
    Augment_DRR(os.path.join(datasets_path, 'xr2ulna', 'trainB' if train else 'testB'), f_name,j,seed_n)
    os.remove(os.path.join(datasets_path, 'xr2ulna', 'trainB' if train else 'testB', f_name))

    # xr2radius
    shutil.copy(os.path.join(data_path, 'DRR', 'Input', f_name),
                os.path.join(datasets_path, 'xr2radius', 'trainA' if train else 'testA'))
    Augment_DRR(os.path.join(datasets_path, 'xr2radius', 'trainA' if train else 'testA'), f_name,j,seed_n)
    os.remove(os.path.join(datasets_path, 'xr2radius', 'trainA' if train else 'testA',f_name))

    shutil.copy(os.path.join(data_path, 'DRR', 'Radius', f_name),
                os.path.join(datasets_path, 'xr2radius', 'trainB' if train else 'testB'))
    Augment_DRR(os.path.join(datasets_path, 'xr2radius', 'trainB' if train else 'testB'), f_name,j,seed_n)
    os.remove(os.path.join(datasets_path, 'xr2radius', 'trainB' if train else 'testB', f_name))


    # xr2ulna_n_radius
    shutil.copy(os.path.join(data_path, 'DRR', 'Input', f_name),
                os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainA' if train else 'testA'))
    Augment_DRR(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainA' if train else 'testA'), f_name,j,seed_n)
    os.remove(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainA' if train else 'testA', f_name))

    shutil.copy(os.path.join(data_path, 'DRR', 'Ulna', f_name),
                os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB1' if train else 'testB1'))
    Augment_DRR(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB1' if train else 'testB1'), f_name,j,seed_n)
    os.remove(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB1' if train else 'testB1', f_name))

    shutil.copy(os.path.join(data_path, 'DRR', 'Radius', f_name),
                os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB2' if train else 'testB2'))
    Augment_DRR(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB2' if train else 'testB2'), f_name,j,seed_n)
    os.remove(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB2' if train else 'testB2', f_name))


if __name__ == '__main__':
    data_path  = r'C:\Users\micha\PycharmProjects\CT_DRR\Data'
    datasets_path  = r'C:\Users\micha\PycharmProjects\CT_DRR\Datasets'
    # remove_and_create(os.path.join(datasets_path))
    dir_content = lambda x,y: sorted(os.listdir(os.path.join(data_path, x,y)))

    input_dir =  dir_content('DRR','Input')
    ulna_dir =   dir_content('DRR','Ulna')
    radius_dir = dir_content('DRR','Radius')
    xr_dir     = dir_content('X-Ray','')

    print (len(input_dir),len(ulna_dir),len(radius_dir))
    assert (len(input_dir)==len(ulna_dir)==len(radius_dir))
    n1 = len(input_dir)
    n2 = len(xr_dir)
    permutated_input_indexes = np.random.permutation(n1)
    permutated_xr_indexes = np.random.permutation(n2)


    # xr2ulna

    # xr2radius
    remove_and_create(os.path.join(datasets_path, 'xr2ulna'))
    remove_and_create(os.path.join(datasets_path, 'xr2ulna', 'trainA'))
    remove_and_create(os.path.join(datasets_path, 'xr2ulna', 'trainB'))
    remove_and_create(os.path.join(datasets_path, 'xr2ulna', 'testA'))
    remove_and_create(os.path.join(datasets_path, 'xr2ulna', 'testB'))

    # xr2radius
    remove_and_create(os.path.join(datasets_path, 'xr2radius'))
    remove_and_create(os.path.join(datasets_path, 'xr2radius', 'trainA'))
    remove_and_create(os.path.join(datasets_path, 'xr2radius', 'trainB'))
    remove_and_create(os.path.join(datasets_path, 'xr2radius', 'testA'))
    remove_and_create(os.path.join(datasets_path, 'xr2radius', 'testB'))

    # xr2ulna_n_radius
    remove_and_create(os.path.join(datasets_path, 'xr2ulna_n_radius'))
    remove_and_create(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainA'))
    remove_and_create(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB1'))
    remove_and_create(os.path.join(datasets_path, 'xr2ulna_n_radius', 'trainB2'))
    remove_and_create(os.path.join(datasets_path, 'xr2ulna_n_radius', 'testA'))
    remove_and_create(os.path.join(datasets_path, 'xr2ulna_n_radius', 'testB1'))
    remove_and_create(os.path.join(datasets_path, 'xr2ulna_n_radius', 'testB2'))

    # train sets
    for j in prange(12):
        print('DRR train set, augment ',j,'/12: ')
        for i in tqdm(permutated_input_indexes[:int(n1*0.9)]):
            f_name = input_dir[i]
            create_datasets(data_path,datasets_path,f_name,j,train=True)

        # test sets
        print('DRR test set, augment ',j,'/12: ')
        for i in tqdm(permutated_input_indexes[int(n1*0.9):]):
            f_name = input_dir[i]
            create_datasets(data_path,datasets_path,f_name,j, train=False)
            # xr2ulna

