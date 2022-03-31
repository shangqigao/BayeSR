'''
Created on Apr 30, 2020

@author: Shangqi Gao
'''
import sys
sys.path.append('../../')
import os 
import time
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage import io, color
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as PSNR

from ssnet_model import SmSpNet as SSNet

class Tester(SSNet):
    def __init__(self):
        SSNet.__init__(self, FLAGS)

    def read_image(self, lr_path, gt_path):
        """
        Args;
            lr_path: The path of a lr image
            gt_path: The path of ground truth
        Returns:
            lr_img: an image, narray, float32
            gt_img: a label, narray, float32
        """
        if FLAGS.task == 'DEN':
            gt_img = io.imread(gt_path).astype(np.float32) / 255.
            if FLAGS.in_channel == 1:
                gt_img = np.expand_dims(gt_img, axis=2)
            np.random.seed(seed=1234)
            sigma = np.sqrt((FLAGS.sigma_read/255)**2 + (FLAGS.sigma_shot/255)*gt_img)
            noise = np.random.normal(size=gt_img.shape)*sigma
            lr_img = gt_img + noise
        else:
            lr_img = io.imread(lr_path) / 255.
            if len(lr_img.shape) == 2:
                lr_img = np.stack([lr_img]*3, axis=2)
            sigma = np.sqrt((FLAGS.sigma_read/255)**2 + (FLAGS.sigma_shot/255)*lr_img)
            lr_img += np.random.normal(size=lr_img.shape)*sigma
            if gt_path is not None:
                gt_img = np.asarray(Image.open(gt_path).convert('RGB')) / 255.
            else:
                gt_img = cv2.resize(lr_img, dsize=(0, 0), fx=FLAGS.upscale, fy=FLAGS.upscale, interpolation=cv2.INTER_LINEAR)
            if len(gt_img.shape) == 2:
                gt_img = np.stack([gt_img]*3, axis=2)
            
        return lr_img, gt_img
 
    def run_test(self):
        test_abs_path = '/home/gaoshangqi/Restoration'
        test_lr_dir = {'DEN': '{}/{}'.format(FLAGS.input_data_dir, FLAGS.dataset),
                       'BiSR': '{}/{}/LR_bicubic/X{}'.format(FLAGS.input_data_dir, FLAGS.dataset, FLAGS.upscale),
                       'ReSR': '{}/{}/LR_mild/X{}'.format(FLAGS.input_data_dir, FLAGS.dataset, FLAGS.upscale),
                       'RWSR': '{}/{}'.format(FLAGS.input_data_dir, FLAGS.dataset)
                        }[FLAGS.task]
        img_mode = 'Gray' if FLAGS.in_channel == 1 else 'RGB'
        test_sr_dir = FLAGS.test_sr_dir
        
        lr_names = sorted(os.listdir(os.path.join(test_abs_path, test_lr_dir)))
        lr_names = [n for n in lr_names if '.png' in n]
        sr_names = sorted(os.listdir(os.path.join(test_abs_path, test_sr_dir)))
        sr_names = [n for n in sr_names if '.png' in n]
        if FLAGS.sample_num == -1:
            samples = np.arange(0, len(lr_names))
        else:
            samples = np.arange(FLAGS.sample_num - 1, FLAGS.sample_num)
        with tf.Graph().as_default():
            sr_pl = tf.placeholder(tf.float32, shape=(3, None, None, 1))
            with tf.variable_scope('SmSpNet', reuse=tf.AUTO_REUSE):
                output = self.Downsampling(sr_pl, ch_to_batch=False)
            ssnet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/Generator')
            ssnet_saver = tf.train.Saver(ssnet_vars)
            sess = tf.Session()
            ssnet_saver.restore(sess, FLAGS.ssnet_checkpoint)
            psnr = []
            for number in tqdm(samples):    
                lr_path = os.path.join(test_abs_path, test_lr_dir, lr_names[number])
                sr_path = os.path.join(test_abs_path, test_sr_dir, sr_names[number])
                lr, sr = self.read_image(lr_path, sr_path)
                img = np.expand_dims(np.transpose(sr, [2, 0, 1]), 3)
                feed_dict = {sr_pl : img}
                out = sess.run(output, feed_dict)
                out = np.transpose(np.squeeze(out), [1, 2, 0])
                psnr.append(PSNR(out, lr, data_range=1.0))
            num = len(psnr)
            mean = sum(psnr) / num
            print(f'mean LRPSNR of {num} images from {FLAGS.dataset}: {mean:0.2f}', )
            
def main(_):
    test = Tester()
    test.run_test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['DEN', 'BiSR', 'ReSR', 'RWSR'], required=True,
                        help='Image restoration task'
                        )
    parser.add_argument('--train_type', choices=['supervised', 'pseudosupervised', 'unsupervised'],
                        default='supervised',
                        help='which kind of training strategies to use'
                        )
    parser.add_argument('--up_type', choices=['interpolation', 'subpixel', 'transpose'],
                        default='subpixel',
                        help='which upsampling operator is used'
                        )
    parser.add_argument('--GAN_type', choices=['GAN', 'LSGAN', 'WGAN', 'WGAN-gp', None], default=None,
                        help='which type of GAN to use, if none, do not use GAN for training'
                        )
    parser.add_argument('--dataset', type=str, default='Set5',
                        help='Test dataset'
                        )
    parser.add_argument('--sample_num', type=int, default=-1,
                        help='which image to super-resolve, -1 denotes all data'
                        )
    parser.add_argument('--repeat_num', type=int, default=1,
                        help='the number of repeatedly sampling number'
                        )
    parser.add_argument('--input_data_dir', type=str, default='./data',
                        help='Directory of test datasets'
                        )
    parser.add_argument('--test_sr_dir', type=str, default='./results',
                        help='Directory of saving reconstructions'
                        )
    parser.add_argument('--ssnet_checkpoint', type=str, default='./model',
                        help='Path of pre-trained RONet model'
                        )
    parser.add_argument('--in_channel', type=int, default=1,
                        help='output channels, 1 for grayscale and 3 for rgb'
                        )
    parser.add_argument('--d_cnnx', type=int, default=3,
                        help='The depth of CNN_x'
                        )
    parser.add_argument('--d_cnnz', type=int, default=3,
                        help='The depth of CNN_z'
                        )
    parser.add_argument('--d_cnnm', type=int, default=3,
                        help='The depth of CNN_m'
                        )
    parser.add_argument('--sigma_read', type=float, default=0,
                        help='image read noise level'
                        )
    parser.add_argument('--sigma_shot', type=float, default=0,
                        help='image shot noise level'
                        )
    parser.add_argument('--train_patch_size', type=int, default=32,
                        help='The batch size for training'
                        )
    parser.add_argument('--regularizer', choices=['l2', 'l1'],
                        help='Which kind of regularizer to use'
                        )
    parser.add_argument('--upscale', type=int, default=1,
                        help='upscaling factor'
                        )
    parser.add_argument('--ensemble', action='store_true',
                        help='If set, use data ensemble'
                        )
    parser.add_argument('--GPU_ids', type=str, default = '0',
                        help = 'Ids of GPUs'
                        )
    FLAGS, unparsed = parser.parse_known_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(FLAGS.GPU_ids)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
