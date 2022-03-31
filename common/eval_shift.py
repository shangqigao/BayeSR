#!/usr/bin/env python
import sys
import os
import os.path
import random
import numpy as np

from PIL import Image
from multiprocessing import Pool
import scipy.misc
#from skimage.measure import structural_similarity as ssim
from common.myssim import compare_ssim as ssim
from common.util import ProgressBar

SCALE = 4
SHIFT = 40
SIZE = 30

def output_measures(img_orig, img_out):
    h, w, c = img_orig.shape
    h_cen, w_cen = int(h / 2), int(w / 2)
    h_left = h_cen - SIZE
    h_right = h_cen + SIZE
    w_left = w_cen - SIZE
    w_right = w_cen + SIZE

    im_h = np.zeros([1, SIZE * 2, SIZE * 2, c])
    im_h[0, :, :, :] = img_orig[h_left:h_right, w_left:w_right, :]
    ssim_h = np.squeeze(im_h)
    im_shifts = np.zeros([(2 * SHIFT + 1) * (2 * SHIFT + 1), SIZE * 2, SIZE * 2, c])
    ssim_shifts = np.zeros([(2 * SHIFT + 1) * (2 * SHIFT + 1), c])
    for hei in range(-SHIFT, SHIFT + 1):
        for wid in range(-SHIFT, SHIFT + 1):
            tmp_l = img_out[h_left + hei:h_right + hei, w_left + wid:w_right + wid, :]
            mean_l = np.mean(tmp_l)
            mean_o = np.mean(img_orig[h_left:h_right, w_left:w_right, :])
            im_shifts[(hei + SHIFT) * (2 * SHIFT + 1) + wid + SHIFT, :, :, :] = tmp_l/mean_l*mean_o

        #ssim_h = np.squeeze(im_h)
            ssim_h = ssim_h.astype('uint8')
            ssim_l = tmp_l.astype('uint8')
            if abs(hei) % 2 == 0 and abs(wid) % 2 == 0:
                for i in range(c):
                    ssim_shifts[(hei + SHIFT) * (2 * SHIFT + 1) + wid + SHIFT, i] \
                        = ssim(ssim_l[:, :, i], ssim_h[:, :, i], gaussian_weights=True, use_sample_covariance=False)

    squared_error = np.square(im_shifts / 255. - im_h / 255.)
    mse = np.mean(squared_error, axis=(1, 2, 3))
    psnr = 10 * np.log10(1.0 / mse)
    return max(psnr), max(np.mean(ssim_shifts, axis=1))

def _open_img_measures(img_p):
    F = np.asarray(Image.open(img_p))
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = 6+SCALE 
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F

def compute_measures(ref_path, res_path):
    psnr, ssim = output_measures(_open_img_measures(ref_path),
                                 _open_img_measures(res_path)
                                )
    img_name = res_path.split('/')[-1]
    process_info = f'Processing {img_name} ...'
    return process_info, psnr, ssim


# as per the metadata file, input and output directories are the arguments

def multiprocess(ref_paths, res_paths, n_threads=4):
    pbar = ProgressBar(len(ref_paths))
    pool = Pool(n_threads)
    outputs = []
    def callback(args):
        outputs.append(args)
        return pbar.update(args[0])
    for (ref_path, res_path) in zip(ref_paths, res_paths):
        pool.apply_async(compute_measures, args=(ref_path, res_path), callback=callback)
    pool.close()
    pool.join()

    psnr = np.mean([x[1] for x in outputs])
    ssim = np.mean([x[2] for x in outputs])
    print(f'All processes done. avg_psnr:{psnr:0.2f}  avg_ssim:{ssim:0.4f}')
    return outputs

def multi_compute(ref_dir, res_dir):
    ref_pngs = sorted([p for p in os.listdir(ref_dir) if p.lower().endswith('png')])
    res_pngs = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('png')])
    ref_paths = [os.path.join(ref_dir, x) for x in ref_pngs]
    res_paths = [os.path.join(res_dir, x) for x in res_pngs]
    if len(ref_pngs) != len(res_pngs):
        raise Exception('Expected equal numbers of images, but got %d and %d'%(len(res_pngs), len(ref_pngs)))
    outputs = multiprocess(ref_paths, res_paths)
    Logpath = os.path.join(res_dir, 'psnr_ssim.txt')
    if os.path.exists(Logpath):
        os.remove(Logpath)
    with open(Logpath, 'a') as f:
        for x in outputs:
            f.write('%s    %0.02f    %0.04f \n' % (x[0], x[1], x[2]))
