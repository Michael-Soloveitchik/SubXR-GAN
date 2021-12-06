import os
import imageio as io
import cv2
from tqdm import tqdm
models_dict = {
    # 'Cycle_GAN_100':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete',
    # 'Cycle_GAN_1000':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete_unet128_100_lambda_b_40',
    # 'Cycle_GAN_0100':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete_unet128_010_lambda_b_40',
    'Cycle_GAN_0099':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete_unet128_009_lambda_b_40',
    'Cycle_GAN_0500':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete_unet128_050_lambda_b_100'
}
def create_GIF(path, GIFs_path=''):
    if not GIFs_path:
        images_path = os.path.join(path, 'web','images')
        GIFs_path = os.path.join(path, 'web','GIFs')
    indexes = list({int(s.split('epoch')[1].split('_')[0]) for s in os.listdir(images_path)})
    gif_packs = {i:[None,None] for i in indexes}
    for f in tqdm(sorted(os.listdir(images_path))):
        i = int(f.split('epoch')[1].split('_')[0])
        if 'real_A' in f:
            gif_packs[i][0] = cv2.imread(os.path.join(images_path, f))
        elif 'fake_B' in f:
            gif_packs[i][1] = cv2.imread(os.path.join(images_path, f))

    os.path.exists(GIFs_path) or os.makedirs(GIFs_path)
    for GIF_i, GIF_arr in tqdm(gif_packs.items()):
        print(GIF_i)
        io.mimsave(os.path.join(GIFs_path, 'gif_epoch' + '_%3d.gif' % (GIF_i)), GIF_arr, duration=1.5)

if __name__ == '__main__':
    for k,p in models_dict.items():
        create_GIF(p)