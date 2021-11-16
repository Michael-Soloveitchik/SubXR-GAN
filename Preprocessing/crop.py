import cv2
import os

def crop(path):
    im = cv2.imread(path)
    m, n, _ = im.shape
    K = 400
    m_new = max(((m - K) // 2), 0)
    m_m_new = max(((m - K) // 2) - 40, 0)
    n_new = max(((n - K) // 2), 0)
    n_n_new = max(((n - K) // 2) - 200, 0)
    im_new = im[m_new:m_new + K, n_n_new:n_n_new + K, :]
    im_new = cv2.resize(im_new, [512, 512])
    cv2.imwrite(path, im_new)