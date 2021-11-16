import shutil
import os
import cv2
import sys

remove_and_create = lambda x: (not shutil.rmtree(x, ignore_errors=True)) and os.makedirs(x)
if __name__ == '__main__':
    DATA_apth  = ' '.join(sys.argv[1:])

    datasets_path = r'C:\Users\micha\PycharmProjects\CT_DRR\Datasets'
    data_path = r'C:\Users\micha\PycharmProjects\CT_DRR\Data'
    remove_and_create(os.path.join(datasets_path, 'xr_real'))
    for i,f in enumerate(os.listdir(DATA_apth)):
        im = cv2.imread(os.path.join(DATA_apth,f))
        m,n,_ = im.shape
        m_new = max(((m-512)//2),0)
        m_m_new = max(((m-512)//2)-40,0)
        n_new = max(((n-512)//2),0)
        n_n_new = max(((n-512)//2)-40,0)
        im_new = im[m_new:m_new+512,n_n_new:n_n_new+512,:]
        print(im_new.shape)
        if (im_new.shape==(512,512,3)):
            cv2.imwrite(os.path.join(data_path, 'X-Ray','Input_'+str(i)+'.jpg'),im)
