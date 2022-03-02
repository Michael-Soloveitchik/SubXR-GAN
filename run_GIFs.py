import os
import imageio as io
import cv2
from tqdm import tqdm
models_dict = {
    # 'Cycle_GAN_100':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete',
    # 'Cycle_GAN_1000':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete_unet128_100_lambda_b_40',
    # 'Cycle_GAN_0100':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete_unet128_010_lambda_b_40',
    # 'Cycle_GAN_0099':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete_unet128_009_lambda_b_40',
    # 'Cycle_GAN_0500':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete_multiscale_2_resnet_1blocks',
    # 'Cycle_GAN_05001':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\drr_complete_2_xr_complete_multiscale_2_global_gamma_100_identity_100_resnet_1blocks_resnet_3blocks',
    'Cycle_GAN_05002':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\batch_30_pyramids_0_35_identity_25_5_unet_16_resnet_1blocks_resnet_3blocks_gamma_45_lambda_A_80',
    # 'Cycle_GAN_05003':r'C:\Users\micha\Research\SubXR-GAN\Models\cycle_gan\SAMPLEs\batch_25_pyramids_0_35_identity_5_5_unet_16_resnet_1blocks_resnet_1blocks_gamma_10_lp_gamma_0_hp_gamma_10_lambda_A_60'
}
def create_GIF(path, GIFs_path=''):
    if not GIFs_path:
        images_path = os.path.join(path, 'web','images')
        GIFs_path = os.path.join(path, 'web','GIFs')
    indexes = list({int(s.split('epoch')[1].split('_')[0]) for s in os.listdir(images_path)})
    gif_packs = {i:[None,None] for i in indexes}
    gif_packs_highpass = {i:[None,None] for i in indexes}
    for f in tqdm(sorted(os.listdir(images_path))):
        i = int(f.split('epoch')[1].split('_')[0])
        if f[:-4].endswith('real_test_A'):
            im = cv2.imread(os.path.join(images_path, f))
            gif_packs[i][0] = im #cv2.imread(os.path.join(images_path, f))
            gif_packs_highpass[i][0] = im-(cv2.pyrUp(cv2.pyrUp(cv2.pyrDown(cv2.pyrDown(im))))+0.)
        elif f[:-4].endswith('fake_test_B'):
            im = cv2.imread(os.path.join(images_path, f))
            gif_packs[i][1] = im #cv2.imread(os.path.join(images_path, f))
            gif_packs_highpass[i][1] = im-(cv2.pyrUp(cv2.pyrUp(cv2.pyrDown(cv2.pyrDown(im))))+0.)

    os.path.exists(GIFs_path) or os.makedirs(GIFs_path)
    for GIF_i in tqdm(gif_packs.keys()):
        GIF_arr = gif_packs[GIF_i]
        GIF_arr_highpass = gif_packs_highpass[GIF_i]
        # print(GIF_i)
        io.mimsave(os.path.join(GIFs_path, 'gif_epoch' + '_%3d.gif' % (GIF_i)), GIF_arr, duration=1.5)
        io.mimsave(os.path.join(GIFs_path, 'gif_highpass_epoch' + '_%3d.gif' % (GIF_i)), GIF_arr_highpass, duration=1.5)

if __name__ == '__main__':
    print(models_dict.values())
    for k,p in models_dict.items():
        print('Now proccessing: ', p)
        create_GIF(p)