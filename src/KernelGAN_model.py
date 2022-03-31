'''
Created on Apr 30, 2020

@author: gsq
'''
import tensorflow as tf
import numpy as np

class KernelGAN:
    def __init__(self, in_channel, downscale):
        # downscale has the form of 2^x
        self.downscale = downscale
        self.strides = 3 if self.downscale == 3 else 2
        self.in_channel = in_channel
        self.initializer = tf.glorot_uniform_initializer()
        self.G_struct = [7, 5, 3, 1, 1, 1]
        self.D_struct = [7, 1, 1, 1, 1, 1]
        self.filters = 64
        self.kernel_size = 3
        self.G_kernel_size = 13
        
    def set_channel(self, mode):
        if mode == 'gray':
            channel = 1
        else:
            channel = 3
        return channel

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

    def Discriminator(self, img_batch, is_training):
        for num in range(len(self.D_struct) - 1):
            img_batch = tf.layers.conv2d(img_batch,
                                         filters=self.filters, kernel_size=self.D_struct[num], strides=1,
                                         kernel_initializer=self.initializer, kernel_regularizer=None,
                                         padding='SAME', use_bias=True, bias_regularizer=None,
                                         trainable=is_training, name='conv{}'.format(num))
            with tf.variable_scope('spectral_normalization{}'.format(num)):
                w = tf.get_default_graph().get_tensor_by_name(img_batch.name)
                w = self.spectral_norm(w)
            if num > 0:
                img_batch = tf.layers.batch_normalization(img_batch)
                img_batch = tf.nn.relu(img_batch)

        img_batch = tf.layers.conv2d(img_batch,
                                     filters=self.in_channel, kernel_size=self.D_struct[-1], strides=1,
                                     kernel_initializer=self.initializer, kernel_regularizer=None,
                                     padding='SAME', use_bias=True, bias_regularizer=None,
                                     trainable=is_training, name='conv{}'.format(len(self.D_struct)-1))
        with tf.variable_scope('spectral_normalization{}'.format(len(self.D_struct)-1)):
            w = tf.get_default_graph().get_tensor_by_name(img_batch.name)
            w = self.spectral_norm(w)
        img_batch = tf.nn.sigmoid(img_batch)

        return img_batch

    def Generator(self, img_batch, is_training):
        '''Generator for implementing an downsampling operator'''
        img_batch = tf.transpose(img_batch, [0, 3, 1, 2])
        shape = img_batch.shape
        img_batch = tf.expand_dims(tf.reshape(img_batch, [-1, shape[2], shape[3]]), axis=3)
        for num in range(len(self.G_struct) - 1):
            img_batch = tf.layers.conv2d(img_batch,
                                         filters=self.filters, kernel_size=self.G_struct[num], strides=1,
                                         kernel_initializer=self.initializer, kernel_regularizer=None,
                                         padding='SAME', use_bias=False, bias_regularizer=None,
                                         trainable=is_training, name='conv{}'.format(num))

        img_batch = tf.layers.conv2d(img_batch,
                                     filters=1, kernel_size=self.G_struct[-1], strides=self.strides,
                                     kernel_initializer=self.initializer, kernel_regularizer=None,
                                     padding='SAME', use_bias=False, bias_regularizer=None,
                                     trainable=is_training, name='conv{}'.format(len(self.G_struct)-1))
        down_shape = img_batch.shape
        img_batch = tf.reshape(img_batch, [shape[0], shape[1], down_shape[1], down_shape[2]])
        img_batch = tf.transpose(img_batch, [0, 2, 3, 1]) 
        return img_batch

    def Downsamplor(self, img_batch, is_training=True, summary=False):
        img_in, img_out = img_batch, []
        with tf.variable_scope('SmSpNet', reuse=tf.AUTO_REUSE):
            for num in range(int(np.log2(self.downscale))):
                with tf.variable_scope('Generator'):
                    img_batch = self.Generator(img_batch, is_training)
                    img_out.append(img_batch)

            with tf.variable_scope('Discriminator'):
                img_real = self.Discriminator(img_in, is_training)

            with tf.variable_scope('Discriminator'):
                img_fake = self.Discriminator(img_out[0], is_training)

        if summary:
            #summarize kernels
            tf.summary.histogram('Downsamplor_input',
                    tf.get_default_graph().get_tensor_by_name('SmSpNet/Generator/conv0/kernel:0'))
            tf.summary.histogram('Downsamplor_output',
                    tf.get_default_graph().get_tensor_by_name('SmSpNet/Discriminator/conv{}/kernel:0'.format(len(self.D_struct)-1)))
            tf.summary.image('input_map', self.ConcatImages([img_in, img_real]), max_outputs=1)
            tf.summary.image('output_map', self.ConcatImages([img_out[0], img_fake]), max_outputs=1)

        if is_training:
            outputs = [img_in, img_real, img_out[0], img_fake]
        else:
            outputs = img_out[-1]

        return outputs

    def calc_curr_k(self):
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/Generator')
        kernels_list = [var for var in vars_list if 'kernel' in var.name and 'Adam' not in var.name and 'beta' not in var.name]
        kernels = [tf.get_default_graph().get_tensor_by_name(var.name) for var in kernels_list]
        delta = tf.ones([1, 1, 1, 1])
        delta = tf.pad(delta, paddings=[[0, 0], [self.G_kernel_size-1, self.G_kernel_size-1], [self.G_kernel_size-1, self.G_kernel_size-1], [0, 0]])
        for i, kernel in enumerate(kernels):
            if i == 0:
                curr_k = tf.nn.conv2d(delta, kernel, strides=(1,1,1,1), padding='VALID')
            else:
                curr_k = tf.nn.conv2d(curr_k, kernel, strides=(1,1,1,1), padding='VALID')
        curr_k = tf.squeeze(curr_k)
        return tf.reverse(curr_k, [0, 1])

    def GANloss(self, output, is_real):
        if is_real:
            label = tf.ones_like(output)
        else:
            label = tf.zeros_like(output)
        return tf.losses.absolute_difference(output, label)

    def DownsampleLoss(self, DownIn, DownOut):
        bicubic_k = [[0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [-.0013275146484375, -0.0039825439453130, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0013275146484380, -0.0039825439453125, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625]]

        def DownBicubic(img, kernel):
            kernel = tf.expand_dims(tf.expand_dims(kernel, axis=2), axis=3)
            img = tf.transpose(img, [0, 3, 1, 2])
            shape = img.shape
            img = tf.expand_dims(tf.reshape(img, [-1, shape[2], shape[3]]), axis=3)
            img = tf.nn.conv2d(img, kernel, strides=(1, self.strides, self.strides, 1), padding='SAME')
            down_shape = img.shape
            img = tf.transpose(tf.reshape(img, [shape[0], shape[1], down_shape[1], down_shape[2]]), [0, 2, 3, 1])
            return img

        def shave(img, ref):
            img_shape = img.shape.as_list()
            ref_shape = img.shape.as_list()
            r, c = max(0, img_shape[1] - ref_shape[1]), max(0, img_shape[2] - ref_shape[2])
            return img[:, r//2:img_shape[1] - r//2 - r%2, c//2:img_shape[2] - c//2 - c%2, :]

        DownIn = DownBicubic(DownIn, bicubic_k)
        DownIn = shave(DownIn, DownOut)
        return tf.losses.mean_squared_error(DownIn, DownOut)

    def SumOfWeightsLoss(self, kernel):
        return tf.abs(tf.constant(1.0) - tf.reduce_sum(kernel))

    def CentralizedLoss(self, kernel):
        shape = kernel.shape.as_list()
        indices = tf.reshape(tf.constant(np.arange(0, shape[0]), dtype=tf.float32), [-1, 1])
        center = shape[0] // 2 + 0.5*(self.strides - shape[0]%2)
        center = tf.constant([center, center])
        r_sum = tf.reshape(tf.reduce_sum(kernel, axis=0), [1, -1])
        c_sum = tf.reshape(tf.reduce_sum(kernel, axis=1), [1, -1])
        r_index = tf.matmul(r_sum, indices) / tf.reduce_sum(kernel)
        c_index = tf.matmul(c_sum, indices) / tf.reduce_sum(kernel)
        return tf.losses.mean_squared_error(tf.stack([tf.squeeze(r_index), tf.squeeze(c_index)]), center)

    def BoundariesLoss(self, kernel):
        def create_penalty_mask(k_size, penalty_scale):
            center_size = k_size // 2 + k_size % 2
            func1 = [np.exp(-z**2/(2*k_size**2))/np.sqrt(2*np.pi*k_size**2) for z in range(-k_size//2+1, k_size//2+1)]
            mask = np.outer(func1, func1)
            mask = 1 - mask/np.max(mask)
            margin = (k_size - center_size) // 2 - 1
            mask[margin:-margin, margin:-margin] = 0
            return penalty_scale*mask
        mask = tf.constant(create_penalty_mask(self.G_kernel_size, 30), dtype=tf.float32)
        label = tf.zeros_like(kernel)
        return tf.losses.absolute_difference(kernel*mask, label)

    def SparsityLoss(self, kernel, power=0.2):
        return tf.losses.absolute_difference(tf.abs(kernel)**power, tf.zeros_like(kernel))

    def loss(self, images, labels=None, name=None):
        '''Define loss functions of training KernelGAN

        '''
        if name == 'Train':
            img_in, img_real, img_out, img_fake = images
            loss_G = self.GANloss(img_fake, is_real=True)
            loss_bicubic = self.DownsampleLoss(img_in, img_out)
            curr_k = self.calc_curr_k()
            tf.summary.image('estimated_kernel', tf.expand_dims(tf.expand_dims(curr_k, axis=0), axis=3), max_outputs=1)
            loss_boundaries = self.BoundariesLoss(curr_k)
            loss_sum2one = self.SumOfWeightsLoss(curr_k)
            loss_centralized = self.CentralizedLoss(curr_k)
            loss_sparse = self.SparsityLoss(curr_k)
            total_loss_G = loss_G + 5.0*loss_bicubic + 0.5*loss_sum2one + 0.5*loss_boundaries \
                    + 0.0*loss_centralized + 0.0*loss_sparse
            loss_D_fake = self.GANloss(img_fake, is_real=False)
            loss_D_real = self.GANloss(img_real, is_real=True)
            total_loss_D = (loss_D_fake + loss_D_real)*0.5
            total_loss = [total_loss_G, total_loss_D]
        else:
            total_loss = tf.losses.mean_squared_error(images, labels)
            tf.summary.scalar(name+'-loss', total_loss)

        return total_loss

    def training(self, loss, learning_rate, global_step):
        '''Set up the training operations
        Create a summarizer to track the loss over time in TensorBoard.
        Create an optimizer and apply gradients to all trainable variables.
    
        The operation returned by this function is what must be passed to
        "sess.run()" call to cause the model to train

        Args:
            loss: The loss needed to optimize
            loss_name: The name of input loss
            learning_rate: the learning rate of current step
            global_step: The current step
        
        Returns:
            train_G: the operation for training Generator.
            train_D: the operation for training Discriminator
        '''
        vars_G = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/Generator')
        optimizer_G = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_G = optimizer_G.minimize(loss[0], var_list=vars_G, global_step=global_step)
        vars_D = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/Discriminator')
        optimizer_D = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_D = optimizer_D.minimize(loss[1], var_list=vars_D, global_step=global_step)
        
        return [train_G, train_D]

    def evaluation(self, images, labels, name):
        '''
        Evaluate the performance of models
        '''
        psnr = tf.reduce_mean(tf.image.psnr(images, labels, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(images, labels, max_val=1.0))
        if name == 'Valid':
            tf.summary.scalar(name+'_psnr', psnr)
            tf.summary.scalar(name+'_ssim', ssim)

        return psnr, ssim

