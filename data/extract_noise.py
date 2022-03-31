"""A script of extracting smooth patches from real-world images
"""
import cv2, os
import numpy as np
from tqdm import tqdm


d = 64 # the size of noise patch
h = 16 # the size of sub-patch
sg = 32 # the strides of extracting noise patch
sl = 16 # the strides of extracting sub-patch
mu = 0.05 # the threshold for the mean of patches
gamma = 0.1 # the threshold for the variance of patches
in_dir = './TrainDatasets/DPED/iphone/train_LR'
out_dir = in_dir + '_noise'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

files = sorted(os.listdir(in_dir))

def smooth_or_not(patch):
    total_mean = np.mean(patch)
    total_var  = np.var(patch)
    row = (patch.shape[0] - h) // sl + 1
    col = (patch.shape[1] - h) // sl + 1
    num_true = 0
    for i in range(row*col):
        up, left = (i // col)*sl, (i % col)*sl
        if len(patch.shape) == 2:
            subpatch = patch[up:up + h, left:left + h]
        else:
            subpatch = patch[up:up + h, left:left + h, :]
        mean = np.mean(subpatch)
        var = np.var(subpatch)
        mean_less = np.abs(total_mean - mean) < mu*total_mean
        var_less = np.abs(total_var - var) < gamma*total_var
        if len(patch.shape) == 2:
            num_true += int(mean_less and var_less)
        else:
            num_true += int(mean_less and var_less)
    return not num_true < row*col

def main():
    num_patch = 0
    for name in tqdm(files):
        path = os.path.join(in_dir, name)
        img = cv2.imread(path) / 255.
        row = (img.shape[0] - d) // sg + 1
        col = (img.shape[1] - d) // sg + 1
        for i in range(row*col):
            up, left = (i // col)*sg, (i % col)*sg
            if len(img.shape) == 2:
                patch = img[up:up + d, left:left + d]
            else:
                patch = img[up:up + d, left:left + d, :]
            smooth = smooth_or_not(patch)
            if smooth:
                cv2.imwrite(os.path.join(out_dir, f'{num_patch:06d}.png'), np.uint8(patch*255.))
                print(f'Saving noise patch {num_patch:06d}.png')
                num_patch += 1

if __name__ == '__main__':
    main()

            
        
        
        
