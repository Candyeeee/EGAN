import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
from sn import spectral_normed_weight
import math


def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name


def get_mat_data(path):
    temp_mat = sio.loadmat(path)
    my_array = np.array([1, 1])
    for key in temp_mat.keys():
        if type(temp_mat[key]) == type(my_array):
            data_key = key
            break
    return temp_mat[data_key]


def metric_mse(im1,im2):
    return tf.reduce_mean(tf.squared_difference(im1,im2))


def grad_x(tensor):
    tensor = tf.pad(tensor, paddings=tf.constant([[0,0],[1,0],[0,0],[0,0]]))
    grad = tensor[:,1:,:,:] - tensor[:,:-1,:,:]
    return grad


def grad_y(tensor):
    tensor = tf.pad(tensor, paddings=tf.constant([[0,0],[0,0],[1,0],[0,0]]))
    grad = tensor[:,:,1:,:] - tensor[:,:,:-1,:]
    return grad


def gen_snconv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.relu, training=True):
    """Define spectral normalization conv for discriminator.
    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.
    Returns:
        tf.Tensor: output
    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'

    fan_in = ksize * ksize * x.get_shape().as_list()[-1]
    fan_out = ksize * ksize * cnum
    stddev = np.sqrt(2. / (fan_in))
    # initializer for w used for spectral normalization
    w = tf.get_variable(name+"_w", [ksize, ksize, x.get_shape()[-1], cnum],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    x = tf.nn.conv2d(x, spectral_normed_weight(w, update_collection=tf.GraphKeys.UPDATE_OPS, name=name+"_sn_w"),
                          strides=[1, stride, stride, 1], dilations=[1, rate, rate, 1], padding=padding, name=name)
    return x


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def downsample(img,step):
    img_down = tf.nn.avg_pool(img,ksize=[1,step,step,1],strides=[1,step,step,1],padding='VALID')
    temp = 0.6*tf.ones(tf.shape(img_down))
    img_down = tf.greater(img_down,temp)
    img_down = tf.cast(img_down,tf.float32)
    return img_down


def bsr(gt_B, netOut_B, mask):
    bsr = 1-tf.reduce_mean(tf.square(gt_B*mask-netOut_B*mask))/tf.reduce_mean(tf.square(gt_B*mask))
    return bsr
    
    
def rmae(gt_B, netOut_B, mask):
    minv = tf.contrib.distributions.percentile(gt_B*mask, 0.1)
    maxv = tf.contrib.distributions.percentile(gt_B*mask, 99.9)
    rmae = tf.reduce_mean(tf.abs(netOut_B*mask-gt_B*mask))/(maxv-minv)
    return rmae
    
    
def psnrS(gt_S, netOut_S, mask):
    minv = np.percentile(gt_S*mask, 0.1)
    maxv = np.percentile(gt_S*mask, 99.9)
    rmae = math.sqrt(np.mean(np.square(netOut_S*mask-gt_S*mask)))
    return 20*math.log10((maxv-minv)/rmae)
    
    

    
