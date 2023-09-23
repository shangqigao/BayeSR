'''
Created on Apr 30, 2020

@author: gsq
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from KernelGAN_model import KernelGAN
import vgg

from common import util

class BayeSR:
    def __init__(self, args):
        self.args = args
        self.pad_mode = 'SYMMETRIC'
        self.initializer = tf.glorot_uniform_initializer()
        self.regularizer = self.set_regularizer(self.args.regularizer)
        self.KernelGAN = KernelGAN(self.args.in_channel, self.args.upscale)
        self.epsilon = np.log(1e10)
        #VGG19 checkpoints
        self.content_layer = 'vgg_19/conv4/conv4_2'
        # blurring HR if true
        self.blurring = self.args.gauss_blur or self.args.motion_blur or self.args.real_blur
        
    def set_regularizer(self, reg_type):
        if reg_type == 'l1':
            regularizer = tf.contrib.layers.l1_regularizer(scale=1e-9)
        elif reg_type == 'l2':
            regularizer = tf.contrib.layers.l2_regularizer(scale=1e-8)
        else:
            regularizer = None
        return regularizer

    def ConcatImages(self, batch_list, columns=2):
        if len(batch_list) // columns != 0:
            batch_list += [tf.zeros_like(batch_list[0])]*(len(batch_list) % columns)
        rows = len(batch_list) // columns
        row_img = []
        for i in range(rows):
            column_img = []
            for j in range(columns):
                column_img.append(batch_list[i*columns+j])
            concat_img = tf.concat(column_img, axis=2)
            row_img.append(concat_img)
        if len(row_img) == 1:
            img = row_img[0]
        else:
            img = tf.concat(row_img, axis=1)

        return img

    def GaussBlur(self, img_batch, kernel_size, sigma1, downscale=1, sigma2=None):
        if sigma2 is None:
            sigma2 = sigma1
        index = np.arange(0, kernel_size) - (kernel_size // 2)
        kx = np.exp(-index**2/(2*sigma1**2))
        ky = np.exp(-index**2/(2*sigma2**2))
        kernel = np.matmul(kx.reshape([-1,1]), ky.reshape([1,-1]))
        kernel = kernel / np.sum(kernel)
        kernel = tf.constant(kernel.reshape([kernel_size, kernel_size, 1, 1]), tf.float32)
        img_batch = tf.transpose(img_batch, [0, 3, 1, 2])
        shape = img_batch.shape
        img_batch = tf.reshape(img_batch, [-1, shape[2], shape[3], 1])
        img_batch = tf.pad(img_batch,
                paddings=[[0,0], [kernel_size//2, kernel_size//2], [kernel_size//2, kernel_size//2], [0,0]],
                mode=self.pad_mode)
        img_batch = tf.nn.conv2d(img_batch, kernel, strides=(1,1,1,1), padding='VALID')
        img_batch = tf.transpose(tf.reshape(img_batch, shape), [0, 2, 3, 1])
        img_batch = img_batch[:, ::downscale, ::downscale, :]

        return img_batch

    def CALayer(self, img_batch, out_chs, reduction=16, is_training=True):
        '''Channel attention layer
        '''
        # global average pooling
        skip = img_batch
        img_batch = tf.reduce_mean(img_batch, axis=(1, 2), keepdims=True)
        img_batch = tf.layers.conv2d(img_batch,
                filters=out_chs // reduction, kernel_size=1, strides=1,
                padding='SAME', use_bias=True, trainable=is_training, name='conv0')
        img_batch = tf.nn.relu(img_batch)
        img_batch = tf.layers.conv2d(img_batch,
                filters=out_chs, kernel_size=1, strides=1,
                padding='SAME', use_bias=True, trainable=is_training, name='conv1')
        img_batch = tf.nn.sigmoid(img_batch)

        return skip*img_batch

    def ResBlock(self, img_batch, out_chs, use_CA=False, is_training=True):
        identity = img_batch
        img_batch = tf.layers.conv2d(img_batch,
                filters=out_chs, kernel_size=self.args.filter_size, strides=1,
                kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                padding='SAME', use_bias=True, bias_regularizer=None,
                trainable=is_training, name='conv0')
        if self.args.BN:
            img_batch = tf.layers.batch_normalization(img_batch, trainable=is_training)
        img_batch = tf.nn.relu(img_batch)

        img_batch = tf.layers.conv2d(img_batch,
                filters=out_chs, kernel_size=self.args.filter_size, strides=1,
                kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                padding='SAME', use_bias=True, bias_regularizer=None,
                trainable=is_training, name='conv1')
        if use_CA:
            with tf.variable_scope('CALayer'):
                img_batch = self.CALayer(img_batch, out_chs, is_training=is_training)
            return img_batch + identity
        else:
            return 0.2*img_batch + identity

    def ConvBlock(self, img_batch, out_chs, is_training=True):
        img_batch = tf.layers.conv2d(img_batch,
                filters=out_chs, kernel_size=self.args.filter_size, strides=1,
                kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                padding='SAME', use_bias=True, bias_regularizer=None,
                trainable=is_training, name='conv0')
        if self.args.BN:
            img_batch = tf.layers.batch_normalization(img_batch, trainable=is_training)
        img_batch = tf.nn.relu(img_batch)

        img_batch = tf.layers.conv2d(img_batch,
                filters=out_chs, kernel_size=self.args.filter_size, strides=1,
                kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                padding='SAME', use_bias=True, bias_regularizer=None,
                trainable=is_training, name='conv1')
        
        return img_batch

    def Upsampling(self, img_batch, num_up, is_training):
        subscale = 3 if self.args.upscale == 3 else 2
        for i in range(num_up):
            with tf.variable_scope('upsample_x{}'.format(subscale**(i+1))):
                if self.args.up_type == 'interpolation':
                    img_batch = tf.image.resize_nearest_neighbor(img_batch,
                            [img_batch.shape[1]*subscale, img_batch.shape[2]*subscale])
                elif self.args.up_type == 'transpose':
                    img_batch = tf.layers.conv2d_transpose(img_batch,
                            filters=self.args.filters, kernel_size=5,
                            strides=subscale, padding='SAME', use_bias=True, trainable=is_training,
                            kernel_initializer=self.initializer, name='conv_transpose')
                else:
                    img_batch = tf.layers.conv2d(img_batch,
                            filters=self.args.filters*subscale**2, kernel_size=self.args.filter_size,
                            strides=1, padding='SAME', use_bias=True, trainable=is_training,
                            kernel_initializer=self.initializer, name='subpixel')
                    img_batch = tf.depth_to_space(img_batch, subscale)

        img_batch = tf.layers.conv2d(img_batch,
                                     filters = self.args.filters, kernel_size=self.args.filter_size, strides=1,
                                     kernel_initializer=self.initializer, kernel_regularizer=None,
                                     padding='SAME', use_bias=True, bias_regularizer=None,
                                     trainable=is_training, name='output1')
        img_batch = tf.layers.conv2d(img_batch,
                                     filters = self.args.in_channel*2, kernel_size=self.args.filter_size, strides=1,
                                     kernel_initializer=self.initializer, kernel_regularizer=None,
                                     padding='SAME', use_bias=True, bias_regularizer=None,
                                     trainable=is_training, name='output2')
        return img_batch

    def Downsampling(self, img_batch, kernel=None):
        if kernel is None:
            for num in range(int(np.log2(self.args.upscale))):
                with tf.variable_scope('Generator'):
                    img_batch = self.KernelGAN.Generator(img_batch, is_training=False)
        else:
            img_batch = util.tensor_batch_degradation(img_batch, kernel, self.args.upscale, padding=True)

        return img_batch

    def ResNet(self, img_batch, d_cnn, is_training):
        '''The CNNs block to infer smooth, sparse and noisy information
        '''
        img_batch = tf.layers.conv2d(img_batch,
                                     filters=self.args.filters, kernel_size=self.args.filter_size, strides=1,
                                     kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                     padding='SAME', use_bias=True, bias_regularizer=None,
                                     trainable=is_training, name='input')
        identity = img_batch

        for num in range(d_cnn):
            with tf.variable_scope('ResBlock{}'.format(num)):
                img_batch = self.ResBlock(img_batch, self.args.filters, self.args.use_CA, is_training)

        return img_batch + identity

    def UNet(self, img_batch, d_cnn, is_training):
        img_batch = img_batch if self.args.upscale == 1 else tf.image.resize_bilinear(img_batch, 
            [img_batch.shape[1]*self.args.upscale, img_batch.shape[2]*self.args.upscale])
        stack, ch, num_pool  = [], self.args.filters, (d_cnn - 2) // 2
        for i in range(num_pool):
            with tf.variable_scope('down_layer{}'.format(i)):
                img_batch = self.ConvBlock(img_batch, ch, is_training)
            stack.append(img_batch)
            img_batch = tf.layers.average_pooling2d(img_batch, 2, 2)
            ch = ch*2

        with tf.variable_scope('bottom'):
            with tf.variable_scope('layer0'):
                img_batch = self.ConvBlock(img_batch, ch, is_training)
            ch = ch // 2
            with tf.variable_scope('layer1'):
                img_batch = self.ConvBlock(img_batch, ch, is_training)

        for i in range(num_pool):
            img_batch = tf.image.resize_bilinear(img_batch,
                    [img_batch.shape[1]*2, img_batch.shape[2]*2])
            img_batch = tf.concat([img_batch, stack.pop()], axis=3)
            ch = ch // 2 if i < num_pool - 1 else ch
            with tf.variable_scope('up_layer{}'.format(i)):
                img_batch = self.ConvBlock(img_batch, ch, is_training)
        
        return img_batch

    def spectral_norm(self, w, iteration=1):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
           """
           power iteration
           Usually iteration = 1 will be enough
           """
           v_ = tf.matmul(u_hat, tf.transpose(w))
           v_hat = tf.nn.l2_normalize(v_)

           u_ = tf.matmul(v_hat, w)
           u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
           w_norm = w / sigma
           w_norm = tf.reshape(w_norm, w_shape)


        return w_norm

    def Discriminator(self, img_batch, is_training, multiscale=False, HR=True, w=1.):
        '''The discriminator used to discriminate the output of encoder
        Args:
            img_batch: a tensor, [batch_size, 64, 64, 16]
        Returns:
            img_batch: a tensor, [batch_size, 1]
        '''
        w = tf.constant(w)
        if HR:
            D_kernels, D_strides, D_filters = [4, 4, 4, 4, 4], [2, 2, 2, 1, 1], [64, 128, 256, 512, 1]
        else: 
            D_kernels, D_strides, D_filters = [4, 4, 4, 4, 4], [2, 1, 1, 1, 1], [64, 128, 256, 512, 1]
        multiscale_batch = []
 
        img_batch = tf.layers.conv2d(img_batch,
                filters=D_filters[0], kernel_size=D_kernels[0], strides=D_strides[0],
                padding='SAME', use_bias=True, trainable=is_training,
                kernel_initializer=self.initializer, name='conv0')
        img_batch = tf.nn.leaky_relu(img_batch, name='act0')

        for i in range(1, len(D_kernels) - 1):
            img_batch = tf.layers.conv2d(img_batch,
                    filters=D_filters[i], kernel_size=D_kernels[i], strides=D_strides[i],
                    padding='SAME', use_bias=False, trainable=is_training,
                    kernel_initializer=self.initializer, name='conv{}'.format(i))
            img_batch = tf.layers.batch_normalization(img_batch, training=is_training, name='bn{}'.format(i))
            img_batch = tf.nn.leaky_relu(img_batch, name='act{}'.format(i))
        multiscale_batch.append(w*img_batch)

        img_batch = tf.layers.conv2d(img_batch, 
                filters=D_filters[-1], kernel_size=D_kernels[-1], strides=D_strides[-1], 
                padding='SAME', use_bias=True, trainable=is_training, 
                kernel_initializer=self.initializer, name='conv{}'.format(len(D_kernels) - 1))
        multiscale_batch.append(w*img_batch)

        if multiscale:
            return multiscale_batch
        else:
            return [w*img_batch]

    def clipslice(self, img_batch, clip=False, start=None, end=None):
        if clip:
            clip_batch = tf.clip_by_value(img_batch, 0.0, 1.0)
        else:
            clip_batch = img_batch
        if start is not None and end is not None:
            slice_batch = clip_batch[start:end, ...]
        else:
            slice_batch = clip_batch

        return slice_batch

    def DxDy(self, I, grad=False):
        #define the difference kernel in horizontal direction
        Dx = tf.expand_dims(tf.expand_dims(tf.constant([[0, 1, -1]], dtype=tf.float32), axis=2), axis=3)

        #define the difference kernel in vertical direction
        Dy = tf.expand_dims(tf.expand_dims(tf.constant([[0], [1], [-1]], dtype=tf.float32), axis=2), axis=3)

        TI = tf.transpose(I, [0, 3, 1, 2])
        RI = tf.expand_dims(tf.reshape(TI, [-1, I.shape[1], I.shape[2]]), axis=3)

        #compute mu_p^T Sigma_p^-1 mu_p
        Ix = tf.pad(RI, paddings=[[0,0], [0,0], [1,1], [0,0]], mode=self.pad_mode)
        Ix = tf.nn.conv2d(Ix, Dx, strides=(1,1,1,1), padding='VALID', name='Dx')
        Iy = tf.pad(RI, paddings=[[0,0], [1,1], [0,0], [0,0]], mode=self.pad_mode)
        Iy = tf.nn.conv2d(Iy, Dy, strides=(1,1,1,1), padding='VALID', name='Dy')
        Ix = tf.reshape(Ix, TI.shape)
        Ix = tf.transpose(Ix, [0, 2, 3, 1])
        Iy = tf.reshape(Iy, TI.shape)
        Iy = tf.transpose(Iy, [0, 2, 3, 1])
        if grad:
            return tf.concat([Ix, Iy], axis=3)
        else:
            return Ix**2 + Iy**2

    def inference(self, img_batch, ker_batch=None, noi_batch=None, lab_batch=None, is_training=True, summary=False):
        '''The inference function for training and test
        Args:
           img_batch: observations of size [batch_size, w, h, channels]
           ker_batch: blur kernels of size [batch_size, k, k, 1]
           noi_batch: estimated noise patches of size [batch_size, w, h, channels]
           lab_batch: references of size [batch_size, w*s, h*s, channels] 
        '''
        #---for test---
        s, d = img_batch.shape.as_list(), 2**((self.args.d_cnnx - 2) // 2)
        h = 0 if s[1] % d == 0 else d - s[1] % d
        w = 0 if s[2] % d == 0 else d - s[2] % d
        img_batch = tf.pad(img_batch, paddings=[[0,0], [0,h], [0,w], [0,0]], mode=self.pad_mode)
        hr_shape = [s[0], s[1]*self.args.upscale, s[2]*self.args.upscale, s[3]]
        #---for test---
        
        with tf.variable_scope('SmSpNet', reuse=tf.AUTO_REUSE):
            real_lr = img_batch
            hr = lab_batch if lab_batch is not None else tf.zeros(hr_shape)
            # generate fake lr from (fake) hr if true
            is_degrading = self.blurring or self.args.train_type != 'supervised'
            if is_training and is_degrading:
                # preprocess if hr is noisy
                if self.args.task in ['SySR', 'ReSR', 'RWSR'] and self.args.train_type == 'unsupervised':
                    hr = self.GaussBlur(hr, 7, 0.5)
                dhr = lab_batch if self.args.task == 'DEN' else self.Downsampling(hr, kernel=ker_batch)
                dhr = self.clipslice(dhr, clip=True)
                fake_lr = self.clipslice(dhr + noi_batch, clip=True)
                y = tf.concat([real_lr, fake_lr], 0) if self.args.train_type != 'supervised' else fake_lr
            else:
                y, dhr = real_lr, real_lr

            uptime = 0 if self.args.setupUnet or self.args.task == 'DEN' else int(np.log2(self.args.upscale))

            # CNN_m
            with tf.variable_scope('cnnm'):
                fm = self.ResNet(y, self.args.d_cnnm, is_training)

            # Estimate mu_m and sigma_m
            with tf.variable_scope('upm'):
                img_batch = self.Upsampling(fm, 0, is_training)
                mum      = img_batch[..., :self.args.in_channel]
                log_varm = img_batch[..., self.args.in_channel:]
                log_varm = tf.clip_by_value(log_varm, -self.epsilon, 0.)

            # Sample m from a Gaussian distribution
            stdm = tf.exp(log_varm / 2)
            m = stdm*tf.random_normal(mum.shape) + mum
            
            # CNN_z
            with tf.variable_scope('cnnz'):
                if not self.args.setupUnet:
                    fz = self.ResNet(y - m, self.args.d_cnnz, is_training)
                    fz = tf.concat([fz, fm], 3)
                else:
                    fz = self.UNet(y - m, self.args.d_cnnz, is_training)
                    fm = tf.image.resize_bilinear(fm, [fz.shape[1], fz.shape[2]])
                    fz = tf.concat([fz, fm], 3)

            # Estimate mu_z and sigma_z
            with tf.variable_scope('upz'):
                img_batch = self.Upsampling(fz, uptime, is_training)
                muz      = img_batch[..., :self.args.in_channel]
                log_varz = img_batch[..., self.args.in_channel:]
                log_varz = tf.clip_by_value(log_varz, -self.epsilon, 0.)

            # Sample z from a Gaussian distribution
            stdz = tf.exp(log_varz / 2)
            z = stdz*tf.random_normal(muz.shape) + muz

            # Estimate mu_w
            log_muw = tf.log(2*self.args.gammaz + 1) - tf.log(muz**2 + tf.exp(log_varz) + 2*self.args.phiz)
            log_muw = tf.stop_gradient(log_muw)

            # Downsample z
            dz = z if self.args.task == 'DEN' else self.Downsampling(z, kernel=ker_batch)

             # CNN_x
            with tf.variable_scope('cnnx'):
                if not self.args.setupUnet:
                    fx = self.ResNet(y - m - dz, self.args.d_cnnx, is_training)
                    fx = tf.concat([fx, fz], 3)
                else:
                    fx = self.UNet(y - m - dz, self.args.d_cnnx, is_training)
                    fx = tf.concat([fx, fz], 3)

            # Estimate mu_x and sigma_x
            with tf.variable_scope('upx'):
                img_batch = self.Upsampling(fx, uptime, is_training)
                mux      = img_batch[..., :self.args.in_channel]
                log_varx = img_batch[..., self.args.in_channel:]
                log_varx = tf.clip_by_value(log_varx, -self.epsilon, 0.)

            # Sample x from a Gaussian distribution
            stdx = tf.exp(log_varx / 2)
            x = stdx*tf.random_normal(mux.shape) + mux

            # Estimate mu_v
            log_muv = tf.log(2*self.args.gammax + 1) - tf.log(self.DxDy(mux) + 4*tf.exp(log_varx) + 2*self.args.phix)
            log_muv = tf.stop_gradient(log_muv)

            # Downsample x
            dx = x if self.args.task == 'DEN' else self.Downsampling(x, kernel=ker_batch)            

            err = y - dx - dz - m
            #err = y - dx - dz

            # Estimate mu_p
            log_mup = tf.log(2*self.args.gamman + 1) - tf.log(err**2 + 2*self.args.phin)
            log_mup = tf.stop_gradient(log_mup)
            
            # Sample n from a Gaussian distribution
            stdn = 1. / tf.exp(log_mup / 2.)
            n = stdn*tf.random_normal(m.shape) + m
            
            if is_training:
                if self.args.train_type != 'supervised':
                    sr = self.clipslice(x + z, clip=True, start=0, end=s[0])
                    dsr = self.clipslice(dx + dz, clip=True, start=0, end=s[0])
                    hr_hat = self.clipslice(x + z, clip=True, start=s[0], end=2*s[0])
                    dhr_hat = self.clipslice(dx + dz, clip=True, start=s[0], end=2*s[0])
                else:
                    sr = self.clipslice(x + z, clip=True)
                    dsr = self.clipslice(dx + dz, clip=True)
                    hr_hat, dhr_hat = sr, dsr
            else:
                if self.args.repeat_num == 1:
                    sr = self.clipslice(mux + muz, clip=True) #for computing PSNR and SSIM
                else:
                    sr = self.clipslice(x + z, clip=True)  #for computing LPIPS and Div. score
                dsr = self.clipslice(dx + dz, clip=True)
                hr_hat, dhr_hat = sr, dsr

            # set up discriminators in LR and HR spaces
            use_HR = self.args.task == 'DEN'
            with tf.variable_scope('Discriminator'):
                with tf.variable_scope('HR'):
                    D_hr_fake = self.Discriminator(hr_hat, is_training, multiscale=False, HR=True, w=1e0)
                with tf.variable_scope('LR'):
                    D_lr_fake = self.Discriminator(dhr_hat, is_training, multiscale=False, HR=use_HR, w=1e0)
                D_fake = D_hr_fake + D_lr_fake
            with tf.variable_scope('Discriminator'):
                with tf.variable_scope('HR'):
                    D_hr_real = self.Discriminator(hr, is_training, multiscale=False, HR=True, w=1e0)
                with tf.variable_scope('LR'):
                    D_lr_real = self.Discriminator(dhr, is_training, multiscale=False, HR=use_HR, w=1e0)
                D_real = D_hr_real + D_lr_real            

        if summary:
            #summarize kernels
            tf.summary.histogram('D_conv3', tf.get_default_graph().get_tensor_by_name('SmSpNet/Discriminator/HR/conv3/kernel:0'))
            #summarize images
            varx, varz, varm = tf.exp(log_varx), tf.exp(log_varz), tf.exp(log_varm)
            tf.summary.scalar('varx', tf.reduce_mean(varx))
            tf.summary.scalar('varz', tf.reduce_mean(varz))
            tf.summary.scalar('varm', tf.reduce_mean(varm))
            muv, muw, mup = tf.exp(log_muv), tf.exp(log_muw), tf.exp(log_mup)
            xhat = self.clipslice(mux, clip=True)
            zhat = self.clipslice(muz, clip=True)
            imgs_list1 = [mux, varx / tf.reduce_max(varx), muv / tf.reduce_max(muv), x]
            imgs_list2 = [muz, varz / tf.reduce_max(varz), muw / tf.reduce_max(muw), z]
            imgs_list3 = [mum, varm / tf.reduce_max(varm), mup / tf.reduce_max(mup), n]
            tf.summary.image('smooth_info', self.ConcatImages(imgs_list1, columns=2), max_outputs=1)
            tf.summary.image('sparse_info', self.ConcatImages(imgs_list2, columns=2), max_outputs=1)
            tf.summary.image('noisy_info', self.ConcatImages(imgs_list3, columns=2), max_outputs=1)
            tf.summary.image('recon_info', self.ConcatImages([xhat, zhat, sr, hr], columns=2), max_outputs=1)
        
        if is_training:
            # Summarize input kernels
            if ker_batch is not None:
                kernels = ker_batch / tf.reduce_max(ker_batch, axis=(1, 2, 3), keepdims=True)
                tf.summary.image('kernels', kernels, max_outputs=1)
            # For computing unsupervised loss of real observations
            output1 = [err, mux, log_varx, muz, log_varz, mum, log_varm, log_muv, log_muw, log_mup]
            #output1 = [self.clipslice(ele, start=0, end=s[0]) for ele in output1]
            # For computing supervised or pseudo-supervised loss
            output2 = [hr_hat, hr, sr, D_fake, D_real]
            outputs = [output1, output2]
        else:
            outputs = sr[:, :self.args.upscale*s[1], :self.args.upscale*s[2], :]


        return outputs

    def GANloss(self, output, is_real):
        '''GAN
        '''
        loss = []
        if is_real:
            if self.args.GAN_type == 'GAN':
               for ele in output:
                    loss.append(-tf.reduce_mean(tf.log(tf.nn.sigmoid(ele) + 1e-10)))
            elif self.args.GAN_type == 'LSGAN':
                for ele in output:
                    loss.append(tf.losses.mean_squared_error(ele, tf.ones_like(ele)))
            elif self.args.GAN_type == 'WGAN':
                for ele in output:
                    loss.append(-tf.reduce_mean(ele))
            else:
                raise ValueError(f'invalid GAN type: {gan}')
        else:
            if self.args.GAN_type == 'GAN':
               for ele in output:
                    loss.append(-tf.reduce_mean(tf.log(1. - tf.nn.sigmoid(ele) + 1e-10)))
            elif self.args.GAN_type == 'LSGAN':
                for ele in output:
                    loss.append(tf.losses.mean_squared_error(ele, tf.zeros_like(ele)))
            elif self.args.GAN_type == 'WGAN':
                for ele in output:
                    loss.append(tf.reduce_mean(ele))
            else:
                raise ValueError(f'invalid GAN type: {gan}')
        return sum(loss) / len(loss)

    def GradPenalty(self, hr, sr):
        '''WGAN-GP
        '''
        s = hr.shape.as_list()
        epsilon = tf.random_uniform(shape=[s[0], 1, 1, 1], minval=0., maxval=1.)
        hrsr = hr + epsilon*(sr - hr)
        with tf.variable_scope('SmSpNet', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Discriminator'):
                D_hrsr = self.Discriminator(hrsr, True)
        grad_D_hrsr = tf.gradients(D_hrsr, [hrsr])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_hrsr), axis=(1, 2, 3)))
        loss = tf.reduce_mean((slopes - 1.)**2)

        return loss

    def loss_xzm(self, inputs):
        '''the variational loss for generative learning
        '''
        err, mux, log_varx, muz, log_varz, mum, log_varm, log_muv, log_muw, log_mup = inputs
        varx, varz, varm = tf.exp(log_varx), tf.exp(log_varz), tf.exp(log_varm)
        muv, muw, mup = tf.exp(log_muv), tf.exp(log_muw), tf.exp(log_mup)
        fidelity = tf.reduce_sum(err**2*mup)
        reg_x = tf.reduce_sum(muv*(self.DxDy(mux) + 4*varx) - log_varx)
        reg_m = tf.reduce_sum(self.args.sigma0*(mum**2 + varm)   - log_varm)
        reg_z = tf.reduce_sum(muw*(muz**2         + varz)   - log_varz)
        loss = fidelity + reg_m + reg_x + reg_z
        #loss = tf.reduce_mean(tf.square(err))
        
        return loss

    def perceptual_loss(self, img_batch, label_batch):
        '''the perceptual loss based on vgg
        '''
        with slim.arg_scope(vgg.vgg_arg_scope()):
            _, end_points_img = vgg.vgg_19(img_batch, num_classes=None, is_training=False)
            _, end_points_label = vgg.vgg_19(label_batch, num_classes=None, is_training=False)
        content_img = end_points_img[self.content_layer]
        content_label = end_points_label[self.content_layer]
        content_loss = tf.losses.absolute_difference(content_label, content_img)
        return content_loss

    def loss(self, ests, labels=None, name=None):
        '''Loss function of training SmSpNet
           loss_net_xzm: for generative learning
           loss_g, loss_d: for generative adversarial learning
           loss_fid: for discriminative learning
        Args:
            prevs: a list, consists of placeholders
            currs: a list, consists of estimations
            labels: a tensor, the ground truth to test
            name: a string, 'Train' or 'Test'
        '''
        if name == 'Train':
            inputs = ests[0]
            hr_hat, hr, sr, D_fake, D_real = ests[1]
            N = tf.reduce_sum(tf.exp(inputs[-1]))
            loss_net_xzm = self.loss_xzm(inputs)
            if self.args.GAN_type is not None:
                loss_g = self.GANloss(D_fake, is_real=True)
                loss_d = self.GANloss(D_fake, is_real=False) + self.GANloss(D_real, is_real=True)
                #loss_p = self.perceptual_loss(hr_hat, hr)
            #loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            if self.args.train_type == 'supervised':
                if self.args.task == 'DEN':
                    loss_fid = tf.reduce_mean(tf.square(sr - labels))
                else:
                    loss_fid = tf.reduce_mean(tf.abs(sr - labels))
                losses = [loss_net_xzm / N + self.args.tau*loss_fid, None]
                #losses = [loss_fid, None] # loss for training baseline models
            else:
                if self.args.task == 'DEN':
                    loss_fid = tf.reduce_mean(tf.square(hr_hat - hr))
                else:
                    loss_fid = tf.reduce_mean(tf.abs(hr_hat - hr))
                if self.args.GAN_type is not None:
                    losses = [loss_net_xzm / N + self.args.tau*loss_fid + self.args.lamda*loss_g, loss_d]
                else:
                    losses = [loss_net_xzm / N + self.args.tau*loss_fid, None]
                    #losses = [loss_fid, None] # loss for training baseline models
        else:
            losses = tf.losses.mean_squared_error(ests, labels) 
        
        return losses
    
    def training(self, losses, learning_rate, global_step):
        '''
        Define training operators
        '''
        #optimizer_xzm = tf.train.GradientDescentOptimizer(learning_rate)
        if self.args.GAN_type is not None:
            optimizer_xzm = tf.train.RMSPropOptimizer(learning_rate)
        else:
            optimizer_xzm = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
        var_net_xzm = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/cnnx') \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/upx') \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/cnnz') \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/upz') \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/cnnm') \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/upm')
        train_xzm = optimizer_xzm.minimize(losses[0], var_list=var_net_xzm, global_step=global_step)

        #optimizer_d = tf.train.GradientDescentOptimizer(learning_rate)
        if self.args.GAN_type is not None:
            optimizer_d = tf.train.RMSPropOptimizer(0.5*learning_rate)
        else:
            optimizer_d = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
        var_d = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/Discriminator')
        if losses[-1] is not None:
            train_d = optimizer_d.minimize(losses[-1], var_list=var_d, global_step=global_step)
        else:
            train_d = tf.constant(1.)

        return [train_xzm, train_d]

    def evaluation(self, images, labels, name):
        '''
        Evaluate the performance of models
        '''
        psnr = tf.reduce_mean(tf.image.psnr(images, labels, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(images, labels, max_val=1.0))
        tf.summary.scalar(name+'_psnr', psnr)
        tf.summary.scalar(name+'_ssim', ssim)
        return psnr, ssim

