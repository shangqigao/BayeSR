'''
Created on Apr 30, 2020

@author: gsq
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')
import argparse
import os
import time
import logging
import tensorflow as tf
import numpy as np

from src.bayesr_model import BayeSR
from src.KernelGAN_model import KernelGAN
from common.load_datasets import Dataloader
from common import util
from args.args_base import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer(BayeSR):
    def __init__(self):
        BayeSR.__init__(self, args)

    def placeholder_inputs(self, batch_size, img_size, ker_size, lab_size):
        """Generate placeholder variables to represent the input tensors
        Args:
            batch_size: The batch size will be bakes into both placeholders.
            
        Returns:
            images_pl: Images placeholder.
            labels_pl: Labels placeholder.
        
        """
        images_pl = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size, self.args.in_channel))
        noises_pl = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size, self.args.in_channel))
        kernel_pl = tf.placeholder(tf.float32, shape=(batch_size, ker_size, ker_size, 1))
        labels_pl = tf.placeholder(tf.float32, shape=(batch_size, lab_size, lab_size, self.args.in_channel))
        return images_pl, kernel_pl, noises_pl, labels_pl

    def fill_feed_dict(self, sess, data_batch, placeholders, feed_dict):
        """Fill the feed_dict for traning the given step
        
        Args:
            data_batch: a list of data to feed
            placeholders: a list of placeholder variables 
            feed_dict: a dictionary
            
        Returns:
            feed_dict: The feed dictionary mapping from placeholder to values
        
        """
        #Create the feed_dict for placeholders
        tensors = sess.run(data_batch, feed_dict=feed_dict)
    
        for i in range(len(tensors)):
            dictionary = {placeholders[i] : tensors[i]}
            feed_dict.update(dictionary)

        return feed_dict

    def do_eval(self, sess, evaluation, feed_dict):
        """Runs one evaluation against the given data
        
        Args:
            sess: The session in which the model has been trained
            evaluation: The evaluation operator
            feed_dcit: The feed dictionary 
        """
        #Run one epoch of evaluation
        psnr, ssim = sess.run(evaluation, feed_dict=feed_dict)
        
        return psnr, ssim

    def run_training(self):
        """Train given data set for a number of steps."""
        #Tell Tensorflow that the model will be built into the default graph
        with tf.Graph().as_default():
            loader = Dataloader(args)
            train_dataset, valid_dataset = loader.generate_dataset(args.input_data_dir)
            train_batch = train_dataset.make_one_shot_iterator().get_next()
            valid_batch = valid_dataset.make_one_shot_iterator().get_next()
            
            #Generate placeholder for images and labels.
            tr_batch_size, va_batch_size, ke_size = args.train_batch_size, args.valid_batch_size, args.kernel_size
            tr_img_size, tr_lab_size = args.train_patch_size, args.train_patch_size*args.upscale
            va_img_size, va_lab_size = args.valid_patch_size, args.valid_patch_size*args.upscale
            train_image_pl, train_kernel_pl, train_noise_pl, train_label_pl = self.placeholder_inputs(tr_batch_size, tr_img_size, ke_size, tr_lab_size)
            valid_image_pl, valid_kernel_pl, valid_noise_pl, valid_label_pl = self.placeholder_inputs(va_batch_size, va_img_size, ke_size, va_lab_size)

            #Add to the graph the operations for setting learning scheduler
            global_step = tf.Variable(0, name='global_step', trainable=False)
            minus_one = tf.assign(global_step, global_step - 1)
            #learning_rates = [1e-3, 1e-4, 1e-5] #SGD
            #learning_rates = [1e-5, 1e-6] #ADAM
            #boundaries = [50000]
            #learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learning_rates)
            learning_rate = tf.train.exponential_decay(1e-4, global_step, args.max_steps//5, decay_rate=0.5, staircase=True)

            #Add to the graph the operators for building networks
            if args.trained_net == 'Downsamplor':
                #Buld a graph to estimate dowmsampling operators for KernelGAN
                train_ests = self.KernelGAN.Downsamplor(train_label_pl, is_training=True, summary=False)
                valid_ests = self.KernelGAN.Downsamplor(valid_label_pl, is_training=False, summary=True)

                #Add to the graph the operations for loss calculation.
                train_loss = self.KernelGAN.loss(train_ests, train_image_pl, name='Train')

                #Add to the graph the operations for training
                train_ops = self.KernelGAN.training(train_loss, learning_rate, global_step)

                #Add to the graph the operations for evaluating models
                valid_loss = self.KernelGAN.loss(valid_ests, valid_image_pl, 'Valid')
                valid_eval = self.KernelGAN.evaluation(valid_ests, valid_image_pl, 'Valid')
            else:
                #Buld a graph to estimate parameters for BayeSR
                if args.gauss_blur or args.motion_blur or args.real_blur:
                    train_kernel, valid_kernel = train_kernel_pl, None
                else:
                    train_kernel, valid_kernel = None, None
                train_ests = self.inference(train_image_pl, train_kernel, train_noise_pl, train_label_pl, is_training=True, summary=False)
                valid_ests = self.inference(valid_image_pl, valid_kernel, valid_noise_pl, valid_label_pl, is_training=False, summary=True)

                if args.train_type == 'supervised':
                    train_loss = self.loss(train_ests, train_label_pl, name='Train')
                else:
                    train_loss = self.loss(train_ests, name='Train')

                #Add to the graph the operations for training
                train_ops = self.training(train_loss, learning_rate, global_step)
                train_loss = [train_loss[0]] + [tf.constant(1.0)] if train_loss[-1] is None else train_loss

                #Add to the graph the operations for evaluating models
                valid_loss = self.loss(valid_ests, labels=valid_label_pl, name='Valid')
                valid_eval = self.evaluation(valid_ests, valid_label_pl, 'Valid')

            #Build the summary tensor based on Tensorflow collection of summaries.
            summary = tf.summary.merge_all()
    
            #Collect needed variables.
            all_vars = tf.global_variables()
            var_dict = {
                'vgg': tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'vgg_19'),
                'BayeSR': tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet'),
                'Downsamplor': tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/Generator')
            }
            var_restore = var_dict[args.trained_net] #+ tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global_step')

            if args.GAN_type is not None:
                var_restore = [v for v in var_restore if 'Adam' not in v.name and 'RMSProp' not in v.name]
            else:
                var_restore = [v for v in var_restore if 'Discriminator' not in v.name and 'Adam' not in v.name and 'RMSProp' not in v.name]
            
            # clip vars for WGAN
            if args.GAN_type == 'WGAN':
                var_clip = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'SmSpNet/Discriminator')
                clip_ops = [v.assign(tf.clip_by_value(v, -0.01, 0.1)) for v in var_clip]

            #Create a saver for an overview of variables, saving and restoring.
            restore_saver = tf.train.Saver(var_restore, max_to_keep=3)
            best_restore_saver = tf.train.Saver(var_restore, max_to_keep=1)
            
            #set the usage of gpu memory
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = False

            #Create a session for running operations on the graph.
            sess = tf.Session(config=config)

            #Instantiate a SummaryWriter to output summaries and graph.
            summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
            
            #calculate the number of trainable variables
            total_paras = 0
            trainable_vars = [var for var in tf.trainable_variables() if var in var_restore]
            for ele in trainable_vars:
                #print(ele.name)
                total_paras += np.prod(np.array(ele.get_shape(), np.int32))
            if total_paras > 1e6:
                total_paras = float(total_paras) / 1e6
                print(f'Total trainable parameters: {total_paras:0.2f}M')
            elif total_paras > 1e3:
                total_paras = float(total_paras) / 1e3
                print(f'Total trainable parameters: {total_paras:0.2f}K')
            else:
                print(f'Total trainable parameters: {total_paras}')

            #Run the operation to initialize variables.
            sess.run(tf.variables_initializer(all_vars))
    
            #Restore checkpoint from disk,
            ckpt_dict = {
                'vgg': args.vgg_checkpoint,
                'BayeSR': args.bayesr_checkpoint,
                'Downsamplor': args.downsamplor_checkpoint
            }
            if args.resume:
                restore_saver.restore(sess, ckpt_dict[args.trained_net])
            # load pre-trained downsamplor for SISR 
            if args.trained_net == 'BayeSR' and len(var_dict['Downsamplor']) > 0:
                downsamplor_saver = tf.train.Saver(var_dict['Downsamplor'])
                downsamplor_saver.restore(sess, ckpt_dict['Downsamplor'])
            # load pre-trained VGG for generative adversarial learning
            if args.GAN_type is not None and len(var_dict['vgg']) > 0:
                vgg_19_saver = tf.train.Saver(var_list=var_dict['vgg'])
                vgg_19_saver.restore(sess, ckpt_dict['vgg'])

            #Start the training loop.
            init_step = sess.run(global_step)
            best_psnr = 0.
            for step in range(init_step + 1, args.max_steps + 1):
                start_time = time.time()
                #Fill a feed dictionary with the actual set of images and labels for this training step.
                feed_dict = {}
                train_placeholders = [train_image_pl, train_kernel_pl, train_noise_pl, train_label_pl]
                feed_dict = self.fill_feed_dict(sess, train_batch, train_placeholders, feed_dict)
                #Update the feed dictionary if corrupted by synthetic kernel and noise
                if args.gauss_blur or args.motion_blur:
                    kernel_batch = util.generate_batch_random_kernel(ke_size, tr_batch_size, args.gauss_blur, 
                                                                     args.motion_blur, args.upscale)
                    feed_dict.update({train_kernel_pl: kernel_batch})
                if args.gauss_blur or args.motion_blur or args.real_blur:
                    noise_batch = util.generate_read_shot_noise(feed_dict[train_image_pl], 12, 4, True)
                    feed_dict.update({train_noise_pl: noise_batch})

                #Save a checkpoint and evaluate the model periodically.
                if step == init_step + 1 or step % 1000 == 0:
                    #save checkpoint
                    checkpoint_file = os.path.join(args.log_dir, 'model')
                    restore_saver.save(sess, checkpoint_file, global_step=step)

                    #update feed dict
                    valid_placeholders = [valid_image_pl, valid_kernel_pl, valid_noise_pl, valid_label_pl]
                    feed_dict = self.fill_feed_dict(sess, valid_batch, valid_placeholders, feed_dict)

                    #Evaluate against the validation set.
                    psnr, ssim = sess.run(valid_eval, feed_dict)
                    if psnr > best_psnr:
                        best_psnr = psnr
                        checkpoint_file = os.path.join(args.log_dir, 'bestmodel')
                        best_restore_saver.save(sess, checkpoint_file)
                    logging.info('Valid evaluation: PSNR=%0.04f  SSIM=%0.04f  Best_PSNR=%0.04f' % (psnr, ssim, best_psnr))

                    #Update the events file
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                #Run one step of the model. The return values are activations from the train_op
                #(which is discarded) and the loss operations.
                if args.GAN_type is not None:
                    d_iters = 1 if step % 500 == 0 or step < 25 else 1
                    for _ in range(d_iters):
                        if args.GAN_type != 'WGAN':
                            sess.run([train_ops[1], minus_one], feed_dict=feed_dict)
                        else:
                            sess.run([train_ops[1], minus_one] + clip_ops, feed_dict=feed_dict)
                        feed_dict = self.fill_feed_dict(sess, train_batch, train_placeholders, feed_dict)

                sess.run(train_ops[0], feed_dict=feed_dict)

                duration = time.time() - start_time
                lr = sess.run(learning_rate)
                #print an overview fairly often.
                if step % 100 == 0:
                    loss1, loss2 = sess.run(train_loss, feed_dict=feed_dict)
                    if args.trained_net == 'BayeSR':
                        #compute means of parameters
                        muv, muw, mup = sess.run(train_ests[0][-3:], feed_dict=feed_dict)
                        muv, muw, mup = np.mean(np.exp(muv)), np.mean(np.exp(muw)), np.mean(np.exp(mup))
                        #Print status
                        logging.info('%s | step:[%7d/%7d] | loss1=%0.04f  | loss2=%0.04f | lr=%1.0e | muv=%0.2e | muw=%0.2e | mup=%0.2e (%0.03f sec)' % \
                                (time.strftime("%Y-%m-%d %H:%M:%S"), step, args.max_steps, loss1, loss2, lr, muv, muw, mup, duration))
                    else:
                        logging.info('%s | step:[%7d/%7d] | loss1=%0.04f  | loss2=%0.04f | lr=%1.0e (%0.03f sec)' % \
                                (time.strftime("%Y-%m-%d %H:%M:%S"), step, args.max_steps, loss1, loss2, lr, duration))
 
def main(_):
    if not args.resume and tf.gfile.Exists(args.log_dir):
        tf.gfile.DeleteRecursively(args.log_dir)
    tf.gfile.MakeDirs(args.log_dir)
    train = Trainer()
    train.run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('BayeSR training and evaluation', allow_abbrev=False)
    add_dataset_args(parser)
    add_experiment_args(parser)
    add_model_args(parser)
    add_hyperpara_args(parser, parser.parse_args())
    args, unparsed = parser.parse_known_args()
    print('--------------------------Arguments-----------------------')
    print(args)
    print('---------------------------End----------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
