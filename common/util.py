'''
Created on Aug 5, 2020

@author: gsq
'''

import sys
import time
import math
from shutil import get_terminal_size

from scipy.ndimage import filters as filters
from scipy.signal import convolve2d
import cv2
import numpy as np
import tensorflow as tf

def generate_gaussian_kernel(k, sigma1=1.6, sigma2=1.6, angle=0):
    """Generate Gaussian kernel`.

    Args:
        k (int): Kernel size.
        sigma1 (float): the first deviation of the Gaussian kernel. Default: 1.6.
        sigma2 (float): the second deviation of the Gaussian kernel. Default: 1.6.
        angle (int): the rotation angle between 0 to 180
    Returns:
        np.array: The Gaussian kernel.
    """
    kernel = np.zeros((k, k))
    # set element at the middle to one, a dirac delta
    kernel[k // 2, k // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    kernel = filters.gaussian_filter(kernel, (sigma1, sigma2))
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0), (k, k))
    return kernel / kernel.sum()

def generate_simple_motionblur_kernel(k, l=13, angle=0):
    """Generate motion kernel`.

    Args:
        k (int): Kernel size.
        l (int): the length of motion kernel, ranges in [5, k]
        angle (int): the rotation angle between 0 to 180
    Returns:
        np.array: The motion kernel.
    """
    kernel = np.zeros((k, k))
    kernel[(k - 1) // 2, (k-l) // 2:(k - l) // 2 + l] = np.ones(l, dtype=np.float32)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0), (k, k))
    return kernel / kernel.sum()

def random_gaussian_kernel(k, scale, minval=0.175, maxval=1.0, minangle=0, maxangle=180):
    """generate random gaussian kernel
    Args:
        k (int): Kernel size.
        scale (int): upscaling factor
        minval (float): the minimal value of sigma
        maxval (float): the maximal value of sigma
        minangle (int): the minimal angle of rotation
        maxangle (int): the maximal angle of rotation
    Returns:
        np.array: a random Gaussian kernel.
    """
    sigma1 = np.random.uniform(scale * minval, scale * maxval)
    sigma2 = np.random.uniform(scale * minval, sigma1)
    angle = np.random.randint(minangle, maxangle)
    kernel = generate_gaussian_kernel(k, sigma1, sigma2, angle)
    return kernel

def random_simple_motionblur_kernel(k, scale, minlen=1, maxlen=5, minangle=0, maxangle=180):
    """generate random motion blur kernel
    Args:
        k (int): Kernel size..
        minlen (int): the minimal length of motion
        maxlen (int): the maximal length of motion
        minangle (int): the minimal angle of rotation
        maxangle (int): the maximal angle of rotation
    Returns:
        np.array: a random motion blur kernel.
    """
    assert k >= scale * maxlen
    length = np.random.randint(scale * minlen, scale * maxlen)
    angle = np.random.randint(minangle, maxangle)
    kernel = generate_simple_motionblur_kernel(k, length, angle)
    return kernel

#----------------------------------------------------------------------------------------------
def random_general_motionblur_kernel(h, w=None):
    """generate random motion blur kernel follows:
       Boracchi et al, Modeling the performance of image restoration from motion blur, TIP 2012.
    Args:
        h (int): height of kernel
        w (int): width of kernel
    Returns:
        k (np.array): a random motion blur kernel of size [h, w].
    """
    w = h if w is None else w
    kdims = [h, w]
    x = randomTrajectory(250)
    k = None
    while k is None:
        k = kernelFromTrajectory(x)

    # center pad to kdims
    pad_width = ((kdims[0] - k.shape[0]) // 2, (kdims[1] - k.shape[1]) // 2)
    pad_width = [(pad_width[0],), (pad_width[1],)]

    if pad_width[0][0] < 0 or pad_width[1][0] < 0:
        k = k[0 : h, 0 : h]
    else:
        k = np.pad(k, pad_width, "constant")
    x1, x2 = k.shape
    if np.random.randint(0, 4) == 1:
        k = cv2.resize(k, (np.random.randint(x1, 5*x1), np.random.randint(x2, 5*x2)), interpolation=cv2.INTER_LINEAR)
        y1, y2 = k.shape
        k = k[(y1 - x1) // 2: (y1 - x1) // 2 + x1, (y2 - x2) // 2: (y2 - x2) // 2 + x2]

    if np.sum(k) < 0.1:
        k = fspecial_gauss(h, 0.1 + 6 * np.random.rand(1))
    k = k / np.sum(k)
    return k

def fspecial_gauss(size, sigma):
    """generate an isotropic gaussian kernel
    Args:
        size (int): kernel size
        sigma (float): standard deviation
    Returns:
        np.array: a gaussian kernel
    """
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def kernelFromTrajectory(x):
    """generate motion blur kernel from given trajectory
    Args:
       x (np.array): given trajectory
    Returns:
       np.array: a motion blur kernel
    """
    h = 5 - np.log(np.random.rand()) / 0.15
    h = np.round(np.min([h, 27])).astype(int)
    h = h + 1 - h % 2
    w = h
    k = np.zeros((h, w))

    xmin = np.min(x[0])
    xmax = np.max(x[0])
    ymin = np.min(x[1])
    ymax = np.max(x[1])
    xthr = np.arange(xmin, xmax, (xmax - xmin) / w)
    ythr = np.arange(ymin, ymax, (ymax - ymin) / h)

    for i in range(1, xthr.size):
        for j in range(1, ythr.size):
            idx = (
                (x[0, :] >= xthr[i - 1])
                & (x[0, :] < xthr[i])
                & (x[1, :] >= ythr[j - 1])
                & (x[1, :] < ythr[j])
            )
            k[i - 1, j - 1] = np.sum(idx)
    if np.sum(k) == 0:
        return
    k = k / np.sum(k)
    k = convolve2d(k, fspecial_gauss(3, 1), "same")
    k = k / np.sum(k)
    return k

def randomTrajectory(T):
    """generate random trajactory
    Args:
       T (int): length of trajectory
    Returns:
       np.array: a random trajectory
    """
    x = np.zeros((3, T))
    v = np.random.randn(3, T)
    r = np.zeros((3, T))
    trv = 1 / 1
    trr = 2 * np.pi / T
    for t in range(1, T):
        F_rot = np.random.randn(3) / (t + 1) + r[:, t - 1]
        F_trans = np.random.randn(3) / (t + 1)
        r[:, t] = r[:, t - 1] + trr * F_rot
        v[:, t] = v[:, t - 1] + trv * F_trans
        st = v[:, t]
        st = rot3D(st, r[:, t])
        x[:, t] = x[:, t - 1] + st
    return x

def rot3D(x, r):
    """perform 3D rotation
    Args:
        x (np.array): input data
        r (float): rotation angle
    Returns:
        np.array: data after rotation
    """
    Rx = np.array([[1, 0, 0], [0, math.cos(r[0]), -math.sin(r[0])], [0, math.sin(r[0]), math.cos(r[0])]])
    Ry = np.array([[math.cos(r[1]), 0, math.sin(r[1])], [0, 1, 0], [-math.sin(r[1]), 0, math.cos(r[1])]])
    Rz = np.array([[math.cos(r[2]), -math.sin(r[2]), 0], [math.sin(r[2]), math.cos(r[2]), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    x = R @ x
    return x
#----------------------------------------------------------------------------------------

def generate_random_kernel(k, gauss=False, motion=False, scale=2):
    """generate random blur kernel
    Args:
        k (int): Kernel size.
        gauss (bool): whether to generate gaussian kernel.
        motion (bool): whether to generate motion blur kernel.
        scale (int): upscaling factor
    Returns:
        np.array: a random kernel.
    """
    if not gauss and not motion:
        kernel = np.zeros([k, k])
        kernel[k // 2, k // 2] = 1.
    elif gauss and motion:
        randnum = np.random.random()
        if randnum < 0.33:
            kernel = random_gaussian_kernel(k, scale)
        elif randnum > 0.66:
            kernel = random_simple_motionblur_kernel(k, scale)
        else:
            kernel = random_general_motionblur_kernel(k)
    elif gauss and not motion:
        kernel = random_gaussian_kernel(k, scale)
    else:
        kernel = random_motionblur_kernel(k, scale)
    return np.expand_dims(kernel.astype(np.float32), axis=2)

def generate_batch_random_kernel(k, b, gauss=False, motion=False, scale=2):
    """generate a batch of random blur kernels
    Args:
        k (int): Kernel size.
        b (int): batch size.
        gauss (bool): whether to generate gaussian kernel.
        motion (bool): whether to generate motion blur kernel.
        scale (int): upscaling factor
    Returns:
        np.array: a batch of random kernels.
    """
    batch = [generate_random_kernel(k, gauss, motion, scale) for _ in range(b)]
    return np.stack(batch)

def generate_read_shot_noise(narray, read=10, shot=0, range_in=False):
    """generate gaussian noise
    Args:
        narray (np.float32): input numpy array.
        read (float): read noise level.
        shot (float): shot noise leve.
        range_in (bool): whether to range betwen 0 and noise level
    Returns:
        np.array: a gaussian noise.
    """
    if range_in:
        read = np.random.uniform(0, read)
        shot = np.random.uniform(0, shot)
    sigma = np.sqrt((read / 255.)**2 + (shot / 255.) * narray)
    noise = np.random.normal(size=narray.shape) * sigma 
    return noise

def tensor_degradation(tensor, kernel, scale=2, padding=False):
    """blurring and downscaling tensor
    Args:
        tensor (tf.float32): input image of shape [h, w, c].
        kernel (tf.float32): blurring kernel of shape [k, k, 1].
        scale (int): downscaling factor.
    Returns:
        tf.float32: degraded image.
    """
    tensor = tf.expand_dims(tf.transpose(tensor, [2, 0, 1]), 3)
    weight = tf.expand_dims(kernel, 3)
    pad = kernel.shape[0] // 2 if padding else 0
    tensor = tf.pad(tensor, paddings=[[0,0], [pad,pad], [pad,pad], [0,0]], mode='SYMMETRIC')
    tensor = tf.nn.conv2d(tensor, weight, strides=(1, scale, scale, 1), padding='VALID', data_format='NHWC')
    tensor = tf.transpose(tensor[..., 0], [1, 2, 0])
    return tensor 

def tensor_batch_degradation(tensor, kernel, scale=2, padding=False):
    """blurring and downscaling tensor
    Args:
        tensor (tf.float32): a batch of images of shape [b, h, w, c].
        kernel (tf.float32): blurring kernel of shape [b, k, k, 1].
        scale (int): downscaling factor.
    Returns:
        tf.float32: a batch of degraded images.
    """
    b, h, w, c = tensor.shape.as_list()
    batch = [tensor_degradation(tensor[i], kernel[i], scale, padding) for i in range(b)]
    return tf.stack(batch)

class ProgressBar(object):
    """A progress bar which can print the progress.

    Modified from:
    https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (
            bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print(f'terminal width is too small ({terminal_width}), '
                  'please consider widen the terminal for better '
                  'progressbar visualization')
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write(f"[{' ' * self.bar_width}] 0/{self.task_num}, "
                             f'elapsed: 0s, ETA:\nStart...\n')
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time + 1e-8
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write(
                '\033[J'
            )  # clean the output (remove extra chars since last display)
            sys.stdout.write(
                f'[{bar_chars}] {self.completed}/{self.task_num}, '
                f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, '
                f'ETA: {eta:5}s\n{msg}\n')
        else:
            sys.stdout.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s, '
                f'{fps:.1f} tasks/s')
        sys.stdout.flush()
