'''
Created on Apr 30, 2020

@author: Shangqi Gao
'''
import sys
sys.path.append('../')
import os 
import time
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage import io, color
from PIL import Image
from scipy.io import loadmat, savemat

from common.eval_noshift import multi_compute as measure_noshift
from common.eval_shift import multi_compute as measure_shift
from skimage.metrics import peak_signal_noise_ratio as PSNR

from bayesr_model import BayeSR
from args.args_base import *

class Tester(BayeSR):
    def __init__(self):
        BayeSR.__init__(self, args)

    def read_image(self, lr_path, gt_path, ke_path):
        """
        Args;
            lr_path: The path of a lr image
            gt_path: The path of ground truth
        Returns:
            lr_img: an image, narray, float32
            gt_img: a label, narray, float32
        """
        np.random.seed(seed=0)
        if args.task == 'DEN':
            gt_img = io.imread(gt_path) / 255.
            if args.in_channel == 1:
                gt_img = np.expand_dims(gt_img, axis=2)
            sigma = np.sqrt((args.sigma_read/255)**2 + (args.sigma_shot/255)*gt_img)
            noise = np.random.normal(size=gt_img.shape)*sigma
            lr_img = np.clip(gt_img + noise, 0., 1.)
        else:
            lr_img = io.imread(lr_path) / 255.
            if len(lr_img.shape) == 2:
                lr_img = np.stack([lr_img]*3, axis=2)
            sigma = np.sqrt((args.sigma_read/255)**2 + (args.sigma_shot/255)*lr_img)
            noise = np.random.normal(size=lr_img.shape)*sigma
            lr_img = np.clip(lr_img + noise, 0., 1.)
            if gt_path is not None:
                gt_img = io.imread(gt_path)
            else:
                gt_img = cv2.resize(lr_img, dsize=(0, 0), fx=args.upscale, fy=args.upscale, interpolation=cv2.INTER_LINEAR)
            if len(gt_img.shape) == 2:
                gt_img = np.stack([gt_img]*3, axis=2)
            if ke_path is not None:
                kernel = loadmat(ke_path)['Kernel']
                kernel = np.expand_dims(kernel, axis=2)
            else:
                kernel = None
            
        return lr_img, gt_img, kernel
    
    def flip(self, image):
        images = [image]
        images.append(image[::-1, :, :])
        images.append(image[:, ::-1, :])
        images.append(image[::-1, ::-1, :])
        images = np.stack(images)
        return images
    
    def mean_of_flipped(self, images):
        image = (images[0] + images[1, ::-1, :, :] + images[2, :, ::-1, :] +
                 images[3, ::-1, ::-1, :])*0.25
        return image
    
    def rotation(self, images):
        return np.swapaxes(images, 1, 2)
 
    def run_test(self, k):
        test_lr_dir = {'DEN': '{}/{}'.format(args.input_data_dir, args.dataset),
                       'BiSR': '{}/{}/LR_bicubic/X{}'.format(args.input_data_dir, args.dataset, args.upscale),
                       'SySR': '{}/{}/LR_degraded/X{}_kernel{}'.format(args.input_data_dir, args.dataset, args.upscale, k),
                       'ReSR': '{}/{}/LR_mild/X{}'.format(args.input_data_dir, args.dataset, args.upscale),
                       'RWSR': '{}/{}'.format(args.input_data_dir, args.dataset)
                        }[args.task]
        test_gt_dir = {'DEN': '{}/{}'.format(args.input_data_dir, args.dataset),
                       'BiSR': '{}/{}/HR'.format(args.input_data_dir, args.dataset),
                       'SySR': '{}/{}/HR'.format(args.input_data_dir, args.dataset),
                       'ReSR': '{}/{}/HR'.format(args.input_data_dir, args.dataset),
                       'RWSR': None
                       }[args.task]
        test_ke_dir = '{}/kernel{}'.format(args.input_kernel_dir, k)
        test_gt_ke_dir = args.input_gt_kernel_dir
        img_mode = 'Gray' if args.in_channel == 1 else 'RGB'
        #test_sr_dir = '{}/{}_SSNet_{}_{}_x{}_read{}_shot{}'.format(args.save_dir, args.dataset, img_mode, args.task, args.upscale, args.sigma_read, args.sigma_shot)
        test_sr_dir = '{}/{}_SSNet_{}_{}_x{}_kernel{}'.format(args.save_dir, args.dataset, img_mode, args.task, args.upscale, k)       
        #load true kernels
        if args.input_gt_kernel_dir is not None:
            gt_kernels = loadmat(os.path.join(test_gt_ke_dir, 'kernels_12.mat'))['kernels']
            gt_kernel = gt_kernels[0, k].astype(np.float64)
            gt_kernel = np.expand_dims(gt_kernel, axis=2)
        else:
            gt_kernel = None
 
        if tf.gfile.Exists(test_sr_dir):
            tf.gfile.DeleteRecursively(test_sr_dir)
        tf.gfile.MakeDirs(test_sr_dir)
        lr_names = sorted(os.listdir(test_lr_dir))
        if test_gt_dir is not None:
            gt_names = sorted(os.listdir(test_gt_dir))
        else:
            gt_names = [None]*len(lr_names)
        if args.input_kernel_dir is not None:
            ke_names = sorted(os.listdir(test_ke_dir))
        else:
            ke_names = [None]*len(lr_names)
        if args.sample_num == -1:
            samples = np.arange(0, len(lr_names))
        else:
            samples = np.arange(args.sample_num - 1, args.sample_num)
        # start to evaluate dataset
        start = time.time()
        with tf.Graph().as_default():
            image_pl = tf.placeholder(tf.float32, shape=(1, 64, 64, args.in_channel))
            output = self.inference(image_pl, is_training=False)
            bayesr_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet')
            bayesr_vars = [v for v in bayesr_vars if 'Discriminator' not in v.name]
            bayesr_saver = tf.train.Saver(bayesr_vars)
            sess = tf.Session()
            bayesr_saver.restore(sess, args.bayesr_checkpoint)
            for number in samples:
                lr_path = os.path.join(test_lr_dir, lr_names[number])
                gt_path = os.path.join(test_gt_dir, gt_names[number]) if gt_names[number] is not None else None
                ke_path = os.path.join(test_ke_dir, ke_names[number]) if ke_names[number] is not None else None 
                gt_name = gt_names[number]
                image, label, kernel = self.read_image(lr_path, gt_path, ke_path)
                shape = image.shape
                image_pl = tf.placeholder(tf.float32, shape=(1, shape[0], shape[1], shape[2]))
                input_images = np.expand_dims(image, 0)
                if kernel is not None:
                    kernel_pl = tf.placeholder(tf.float32, shape=(1, kernel.shape[0], kernel.shape[1], kernel.shape[2]))
                    input_kernels = np.expand_dims(kernel, 0)
                elif gt_kernel is not None:
                    kernel_pl = tf.placeholder(tf.float32, shape=(1, gt_kernel.shape[0], gt_kernel.shape[1], gt_kernel.shape[2]))
                    input_kernels = np.expand_dims(gt_kernel, 0)
                else:
                    kernel_pl, input_kernels = None, None
                output = self.inference(image_pl, kernel_pl, is_training=False)
                if input_kernels is not None:
                    feed_dict = {image_pl : input_images, kernel_pl : input_kernels}
                else:
                    feed_dict = {image_pl : input_images}
                for i in range(args.repeat_num):
                    output_image = sess.run(output, feed_dict)[0]
                    img_name = ''.join(lr_names[number].split('.')[:-1])
                    img_name = img_name + '.png' if args.repeat_num == 1 else img_name + '_sample{:05d}'.format(i) + '.png'
                    sr_img = np.around(output_image*255.0).astype(np.uint8)
                    io.imsave(os.path.join(test_sr_dir, img_name), np.squeeze(sr_img))    
                    print('saving {}'.format(img_name))
        duration = time.time() - start
        mean_dura = duration / (len(gt_names)*args.repeat_num)
        print(f'Avg_reconstruction_time_per_image: {mean_dura:0.2f}')
        if args.task == 'ReSR' and args.repeat_num == 1:
            measure_shift(test_gt_dir, test_sr_dir)
        elif args.task in ['BiSR', 'SySR'] and args.repeat_num == 1:
            measure_noshift(test_gt_dir, test_sr_dir, args.upscale, 'ycbcr')
        elif args.task == 'DEN' and args.repeat_num == 1:
            measure_noshift(test_gt_dir, test_sr_dir, args.upscale, 'rgb')
        else:
            print('Do not evaluate images!')
            
def main(_):
    test = Tester()
    for k in range(12):
        test.run_test(k)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('BayeSR test and evaluation', allow_abbrev=False)
    add_dataset_args(parser)
    add_experiment_args(parser)
    add_model_args(parser)
    add_hyperpara_args(parser, parser.parse_args())
    args, unparsed = parser.parse_known_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
