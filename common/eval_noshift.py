import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage import io, color
from multiprocessing import Pool
from common.util import ProgressBar

from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

def imgcrop(img, scale):
    height = img.shape[0] - img.shape[0] % scale
    width = img.shape[1] - img.shape[1] % scale
    if len(img.shape) == 2:
        img = img[0:height, 0:width]
    else:
        img = img[0:height, 0:width, :]
    return img

def shave(img, scale):
    if len(img.shape) == 2:
        img = img[scale:img.shape[0]-scale, scale:img.shape[1]-scale]
    else:
        img = img[scale:img.shape[0]-scale, scale:img.shape[1]-scale, :]
    return img

def compute_measures(img_path, lab_path, scale, mode= 'rgb'):
    img = io.imread(img_path)
    lab = io.imread(lab_path)
    if mode == 'ycbcr':
        img = color.rgb2ycbcr(img)[:,:,0]
        lab = color.rgb2ycbcr(lab)[:,:,0]
    if scale > 1:
        img = imgcrop(img, scale)
        img = shave(img, scale)
        lab = imgcrop(lab, scale)
        lab = shave(lab, scale)

    if len(lab.shape) == 2:
        psnr = PSNR(img, lab, data_range=255)
        ssim = SSIM(img, lab, data_range=255, multichannel=False)
    else:
        psnr = PSNR(img, lab, data_range=255)
        ssim = SSIM(img, lab, data_range=255, multichannel=True)
    img_name = img_path.split('/')[-1]
    process_info = f'Processing {img_name} ...'

    return process_info, psnr, ssim

def multiprocess(ref_paths, res_paths, scale, mode, n_threads=20):
    pbar = ProgressBar(len(ref_paths))
    pool = Pool(n_threads)
    outputs = []
    def callback(args):
        outputs.append(args)
        return pbar.update(args[0])
    for (ref_path, res_path) in zip(ref_paths, res_paths):
        pool.apply_async(compute_measures, args=(ref_path, res_path, scale, mode), callback=callback)
    pool.close()
    pool.join()

    psnr = np.mean([x[1] for x in outputs])
    ssim = np.mean([x[2] for x in outputs])
    print(f'All processes done. avg_psnr:{psnr:0.2f}  avg_ssim:{ssim:0.4f}')

def multi_compute(ref_dir, res_dir, scale, mode):
    ref_pngs = sorted([p for p in os.listdir(ref_dir)])
    res_pngs = sorted([p for p in os.listdir(res_dir)])
    ref_paths = [os.path.join(ref_dir, x) for x in ref_pngs]
    res_paths = [os.path.join(res_dir, x) for x in res_pngs]
    if len(ref_pngs) != len(res_pngs):
        raise Exception('Expected equal numbers of images, but got %d and %d'%(len(res_pngs), len(ref_pngs)))
    multiprocess(ref_paths, res_paths, scale, mode)

