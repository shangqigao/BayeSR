'''
Created on Apr 30, 2020

@author: gsq
'''
import os, cv2, random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from common import util

class Dataloader:
    def __init__(self, args):
        #if True, use data augmentation
        self.augment = args.augment
        self.task = args.task
        self.upscale = args.upscale
        self.in_channel = args.in_channel
        self.gauss_blur = args.gauss_blur
        self.motion_blur = args.motion_blur
        self.sigma_read = args.sigma_read
        self.sigma_shot = args.sigma_shot
        self.train_type = args.train_type
        #if True, the noise level ranges in [0, sigma]
        self.range = args.range
        self.sample_num = args.sample_num
        #set the training directory of LR images
        self.TRAIN_LR_DIR = {'DEN': 'DIV2K/DIV2K_HR/DIV2K_HR_train',
                             'BiSR': 'DIV2K/DIV2K_LR_bicubic/DIV2K_LR_train/X{}'.format(self.upscale),
                             'SySR': 'DIV2K/DIV2K_LR_bicubic/DIV2K_LR_train/X{}'.format(self.upscale),
                             'ReSR': 'DIV2K/DIV2K_LR_mild/DIV2K_LR_train/X{}'.format(self.upscale),
                             'RWSR': 'DPED/iphone/train_LR'
        }[self.task]
        #set the training directory of blur kernels
        self.TRAIN_KE_DIR = {'DEN': None, 'BiSR': None, 'SySR': None, 'ReSR': None, 'RWSR': None}[self.task]
        #set the training directory of noisy images
        self.TRAIN_NO_DIR = {'DEN': None,
                             'BiSR': None,
                             'SySR': 'DPED/iphone/train_LR_noise',
                             'ReSR': 'DIV2K/DIV2K_LR_mild/DIV2K_LR_train/X{}_noise'.format(self.upscale),
                             'RWSR': 'DPED/iphone/train_LR_noise'
        }[self.task]
        #set the training directory of HR images 
        self.TRAIN_HR_DIR = {'supervised': 'DIV2K/DIV2K_HR/DIV2K_HR_train',
                             'pseudosupervised': 'Flickr2K/Flickr2K_HR',
                             'unsupervised': None
        }[self.train_type]
        #set the validation directory of LR images
        self.VALID_LR_DIR = {'DEN': 'DIV2K/DIV2K_HR/DIV2K_HR_valid',
                             'BiSR': 'DIV2K/DIV2K_LR_bicubic/DIV2K_LR_valid/X{}'.format(self.upscale),
                             'SySR': 'DIV2K/DIV2K_LR_bicubic/DIV2K_LR_valid/X{}'.format(self.upscale),
                             'ReSR': 'DIV2K/DIV2K_LR_mild/DIV2K_LR_valid/X{}'.format(self.upscale),
                             'RWSR': 'DPED/iphone/valid_LR'
        }[self.task]
        #set the validation directory of HR images
        self.VALID_HR_DIR = {'DEN': 'DIV2K/DIV2K_HR/DIV2K_HR_valid',
                             'BiSR': 'DIV2K/DIV2K_HR/DIV2K_HR_valid',
                             'SySR': 'DIV2K/DIV2K_HR/DIV2K_HR_valid',
                             'ReSR': 'DIV2K/DIV2K_HR/DIV2K_HR_valid',
                             'RWSR': None
        }[self.task]
         
        self.train_batch_size = args.train_batch_size 
        self.valid_batch_size = args.valid_batch_size

        self.train_patch_size = args.train_patch_size
        self.valid_patch_size = args.valid_patch_size
        self.train_label_size = args.train_patch_size*self.upscale
        self.valid_label_size = args.valid_patch_size*self.upscale
        
        self.num_data_threads = args.threads
        self.shuffle_buffer_size  = 100

        #The maximal sliding steps for aligning LR and HR paires
        self.align_step = {'DEN': 0, 'BiSR': 0, 'SySR': 0, 'ReSR': 10, 'RWSR':0}[self.task]


    def extract_image(self, input_data_dir, mode):
        '''The function to extract images
        Arg:
            input_data_dir: the dir of dataset
            mode: 'train' or 'validation' or 'test'
        return:
            datatset: A tensor of size [2, num_img, height, width, channels]
        '''
        lr_dir = {'train': os.path.join(input_data_dir, self.TRAIN_LR_DIR),
                  'valid': os.path.join(input_data_dir, self.VALID_LR_DIR)
                  }[mode]
        train_ke_dir = os.path.join(input_data_dir, self.TRAIN_KE_DIR) if self.TRAIN_KE_DIR is not None else None
        ke_dir = {'train': train_ke_dir, 'valid': None}[mode]
        train_no_dir = os.path.join(input_data_dir, self.TRAIN_NO_DIR) if self.TRAIN_NO_DIR is not None else None
        no_dir = {'train': train_no_dir, 'valid': None}[mode]
        train_hr_dir = os.path.join(input_data_dir, self.TRAIN_HR_DIR) if self.TRAIN_HR_DIR is not None else None
        valid_hr_dir = os.path.join(input_data_dir, self.VALID_HR_DIR) if self.VALID_HR_DIR is not None else None
        hr_dir = {'train': train_hr_dir, 'valid': valid_hr_dir }[mode]
        if not tf.gfile.Exists(lr_dir):
            raise ValueError(f'{lr_dir} does not exit.')

        def list_files(d):
            files = sorted(os.listdir(d))
            files = [os.path.join(d, f) for f in files]
            return files
    
        lr_files = list_files(lr_dir)
        hr_files = list_files(hr_dir) if hr_dir is not None else ['None']*len(lr_files)
        ke_files = list_files(ke_dir) if ke_dir is not None else ['None']*len(lr_files)
        no_files = list_files(no_dir) if no_dir is not None else ['None']*len(lr_files)
        total_lr_num = len(lr_files)
        total_ke_num = len([f for f in ke_files if f != 'None'])
        total_no_num = len([f for f in no_files if f != 'None'])
        total_hr_num = len([f for f in hr_files if f != 'None'])
        print(f'{mode}: total lr={total_lr_num}, kernel={total_ke_num}, noise={total_no_num}, hr={total_hr_num}')

        #set the number of sampled training images
        if self.sample_num != -1 and mode == 'train':
            if self.sample_num <= min(total_lr_num, total_hr_num):
                lr_files = lr_files[:self.sample_num]
                hr_files = hr_files[:self.sample_num]
            else:
                raise ValueError(f'sampling numbers are bigger than image numbers')
        sampled_lr_num, sampled_hr_num = len(lr_files), len(hr_files)
        print(f'{mode}: sampled lr={sampled_lr_num}, hr={sampled_hr_num}')
        
        #set the numbers of files to be equal
        max_len = max([len(lr_files), len(ke_files), len(no_files), len(hr_files)])
        lr_files = lr_files*(max_len // len(lr_files)) + lr_files[:max_len % len(lr_files)]
        ke_files = ke_files*(max_len // len(ke_files)) + ke_files[:max_len % len(ke_files)]
        no_files = no_files*(max_len // len(no_files)) + no_files[:max_len % len(no_files)]
        hr_files = hr_files*(max_len // len(hr_files)) + hr_files[:max_len % len(hr_files)]
        kernels = [loadmat(ke_file) for ke_file in ke_files]

        dataset = tf.data.Dataset.from_tensor_slices((lr_files, kernels, no_files, hr_files))

        def _read_image(lr_file, kernel, no_file, hr_file):
            print('Reading images!')
            lr_image = tf.image.decode_image(tf.read_file(lr_file), channels=self.in_channel)
            if ke_dir is not None:
                kernel = tf.constant(kernel, tf.float32)
                kernel = tf.expand_dims(kernel, 2)
            else:
                kernel = tf.constant(np.zeros([25, 25, 1], np.float32), tf.float32)

            if no_dir is not None:
                noise = tf.image.decode_image(tf.read_file(no_file), channels=self.in_channel)
            else:
                noise = tf.zeros([64, 64, self.in_channel], tf.uint8)

            if hr_dir is not None:
                hr_image = tf.image.decode_image(tf.read_file(hr_file), channels=self.in_channel)
            else:
                lr_shape = tf.shape(lr_image)
                hr_image = tf.expand_dims(lr_image, 0)
                hr_image = tf.image.resize_bicubic(hr_image,
                        [lr_shape[0]*self.upscale, lr_shape[1]*self.upscale])
                hr_image = tf.cast(hr_image[0], tf.uint8)

            def _alignment(lr_image, hr_image, step=self.align_step):
                lr_shape = tf.shape(lr_image)
                height = lr_shape[0] - 2*step
                width = lr_shape[1] - 2*step
                hr = tf.expand_dims(hr_image, 0)
                bicubic_lr = tf.image.resize_bicubic(hr, [lr_shape[0], lr_shape[1]])[0]
                initial_lr = tf.slice(bicubic_lr, [step, step, 0], [height, width, -1])
                lr = tf.slice(lr_image, [step, step, 0], [height, width, -1])
                lr = tf.cast(lr, tf.float32)
                mse = tf.reduce_mean(tf.square(lr - initial_lr))
                shift_up = 0
                shift_left = 0
                for row in range(-step, step):
                    for column in range(-step, step):
                        new_lr = tf.slice(bicubic_lr, [step + row, step + column, 0], [height, width, -1])
                        new_lr = tf.cast(new_lr, tf.float32)
                        new_mse = tf.reduce_mean(tf.square(lr - new_lr))
                        def _true_fn():
                            return row, column, new_mse
                        def _false_fn():
                            return shift_up, shift_left, mse
                        shift_up, shift_left, mse = tf.cond(new_mse < mse, true_fn=_true_fn, false_fn=_false_fn)
                return shift_up, shift_left
            
            #In realistic case, data is preprocessed using alignment
            if self.task == 'ReSR':
                shift_up, shift_left = _alignment(lr_image, hr_image)
            else:
                shift_up, shift_left = 0, 0

            return lr_image, kernel, noise, hr_image, shift_up, shift_left
    
        dataset = dataset.map(_read_image,
                              num_parallel_calls=self.num_data_threads,
                              )
        if max_len < 3000:
            dataset = dataset.cache()

        return dataset

    def extract_batch(self, dataset, mode, lr_size, hr_size):
        '''The function to extract sub-images for training
        '''
        if mode == 'train':
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        dataset = dataset.repeat()
    
        def _preprocess(lr, kernel, noise, hr, shift_up, shift_left):
            def _img_downscale(img, upscale):
                scale = tf.random_uniform(shape=[], minval=1., maxval=8./self.upscale, dtype=tf.float32)
                scale = tf.cast(scale, tf.int32)*upscale
                shape = tf.shape(img)
                img = img[0:shape[0]:scale, 0:shape[1]:scale, :]
                return img

            def _flip_rotation(values, fn):
                def _done():
                    return [fn(v) for v in values]
                def _notdone():
                    return values
    
                pred = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)
                values = tf.cond(pred, _done, _notdone)
                return values

            def _img_add_noise(img):
                img = tf.image.convert_image_dtype(img, tf.float32) 
                if self.range and mode == 'train':
                    sigma_read = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)*self.sigma_read / 255.
                    sigma_shot = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)*self.sigma_shot / 255.
                else:
                    sigma_read = (self.sigma_read / 2) / 255.
                    sigma_shot = (self.sigma_shot / 2) / 255.
                sigma = tf.sqrt(sigma_read ** 2 + sigma_shot * img)
                noise = tf.random_normal(shape=tf.shape(img)) * sigma
                img_noise = img + noise
                return tf.cast(tf.clip_by_value(img_noise*255., 0., 255.), tf.uint8)
            
            # Corrupt lr with noise, 
            if self.sigma_read > 0 or self.sigma_shot > 0:
                lr =  _img_add_noise(lr)

            # set hr to be lr if hr is unavailable in training stage
            hr = lr if mode == 'train' and self.TRAIN_HR_DIR is None else hr

            # Crop lr and hr patches       
            lr_shape = tf.shape(lr)
            hr_shape = tf.shape(hr)
            no_shape = tf.shape(noise)
            lr_up   = tf.random_uniform(shape=[], minval=self.align_step, maxval=lr_shape[0] - lr_size - self.align_step, dtype=tf.int32)
            lr_left = tf.random_uniform(shape=[], minval=self.align_step, maxval=lr_shape[1] - lr_size - self.align_step, dtype=tf.int32)
            lr = tf.slice(lr, [lr_up, lr_left, 0], [lr_size, lr_size, -1])

            no_up   = tf.random_uniform(shape=[], minval=0, maxval=no_shape[0] - lr_size, dtype=tf.int32) if lr_size < 64 else 0
            no_left = tf.random_uniform(shape=[], minval=0, maxval=no_shape[1] - lr_size, dtype=tf.int32) if lr_size < 64 else 0
            noise = tf.slice(noise, [no_up, no_left, 0], [lr_size, lr_size, -1])
            
            if mode == 'train' and self.train_type != 'supervised':
                hr_up   = tf.random_uniform(shape=[], minval=0, maxval=hr_shape[0] - hr_size, dtype=tf.int32)
                hr_left = tf.random_uniform(shape=[], minval=0, maxval=hr_shape[1] - hr_size, dtype=tf.int32)
            else:   
                hr_up   = (lr_up + shift_up)*self.upscale
                hr_left = (lr_left + shift_left)*self.upscale
            hr = tf.slice(hr, [hr_up, hr_left, 0], [hr_size, hr_size, -1])

            if mode == 'train' and self.augment:
                lr, hr = _flip_rotation([lr, hr], tf.image.flip_left_right)
                lr, hr = _flip_rotation([lr, hr], tf.image.flip_up_down)
                lr, hr = _flip_rotation([lr, hr], tf.image.rot90)
    
            lr = tf.image.convert_image_dtype(lr, tf.float32)
            hr = tf.image.convert_image_dtype(hr, tf.float32)

            # generate pseudo noise by substracting mean
            noise = tf.image.convert_image_dtype(noise, tf.float32)
            noise -= tf.reduce_mean(noise, axis=(0, 1), keepdims=True)

            return lr, kernel, noise, hr
        
        dataset = dataset.map(
            _preprocess,
            num_parallel_calls=self.num_data_threads,
            )
        batch_size = {
            'train': self.train_batch_size,
            'valid': self.valid_batch_size,
            }[mode]
        drop_remainder = {
            'train': True,
            'valid': True,
            }[mode]
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        return dataset
    
    def generate_dataset(self, input_data_dir):

        train_dataset = self.extract_image(input_data_dir, 'train')
        valid_dataset = self.extract_image(input_data_dir, 'valid')
        train_dataset = self.extract_batch(train_dataset, 'train', self.train_patch_size, self.train_label_size)
        valid_dataset = self.extract_batch(valid_dataset, 'valid', self.valid_patch_size, self.valid_label_size)
        return train_dataset, valid_dataset
