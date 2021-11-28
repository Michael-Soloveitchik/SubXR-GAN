import os
import imageio as io
import cv2
from tqdm import tqdm
models_dict = {
    'Cycle_GAN_100':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete',
    'Cycle_GAN_1000':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete_unet128_100',
    'Cycle_GAN_025':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete_unet128_025'
}
GIFs_dict = {}
for k,p in models_dict.items():

    os.path.exists(os.path.join(p,'web','GIFs')) or os.makedirs(os.path.join(p,'web','GIFs'))
    indexes = list({int(s.split('epoch')[1].split('_')[0]) for s in os.listdir(os.path.join(p, 'web','images'))})
    gif_packs = {i:[None,None] for i in indexes}
    for f in tqdm(sorted(os.listdir(os.path.join(p, 'web','images')))):
        i = int(f.split('epoch')[1].split('_')[0])
        if 'real_A' in f:
            gif_packs[i][0] = cv2.imread(os.path.join(p, 'web','images',f))
        elif 'fake_B' in f:
            gif_packs[i][1] = cv2.imread(os.path.join(p, 'web', 'images', f))
        # GIFs_dict[k] = [[] for i]

    for GIF_i, GIF_arr in tqdm(gif_packs.items()):
        print(GIF_i)
        io.mimsave(os.path.join(os.path.join(p,'web','GIFs'), 'gif_epoch' + '_%3d.gif' % (GIF_i)), GIF_arr, duration=1.5)