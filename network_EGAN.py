import time
import os, random
import re
import glob
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib as tf_contrib
from utils import gen_snconv
from utils import filepath_to_name, grad_x, grad_y, downsample


def random_crop(image, label, mask, edge, crop_h, crop_w):
    if image.shape[0]!=label.shape[0] or image.shape[0]!=label.shape[0]:
        raise Exception('Image and label must have the same dimensions!')
    if (crop_w <= image.shape[1] and crop_h <= image.shape[0]):
        x = random.randint(0, image.shape[1] - crop_w)
        y = random.randint(0, image.shape[0] - crop_h)
        return image[y:y+crop_h, x:x+crop_w], label[y:y+crop_h, x:x+crop_w], mask[y:y+crop_h, x:x+crop_w], edge[y:y+crop_h, x:x+crop_w]
    else:
        raise Exception('crop shape exceeds image dimensions!')


weight_init = tf_contrib.layers.variance_scaling_initializer()


def conv(x, channels, kernel=3, stride=1, padding='SAME', use_bias=True, scope='conv_0', is_training=True):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init, trainable=is_training,
                             #kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)

        return x


def relu(x):
    return tf.nn.relu(x)


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)


def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope):

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = relu(x)

        if downsample:
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')
        else:
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')

        return x + x_init


class GAN(object):
    model_name = "Bone_Suppression_GAN"

    def __init__(self, sess, args):
        self.sess = sess
        self.checkpoints_dir = args.checkpoints_dir
        self.result_dir = args.result_dir
        self.num_epochs = args.num_epochs
        self.batch_sz = args.batch_sz
        self.log_dir = args.summary_dir
        self.crop_h = args.crop_h
        self.crop_w = args.crop_w
        self.train_files = glob.glob(args.train_path + '/*.mat')
        self.val_files = glob.glob(args.val_path + '/*.mat')
        self.train_num = self.train_files.__len__()
        self.val_num = self.val_files.__len__()
        self.num_batches = int(np.floor(self.train_num / self.batch_sz))
        self.d_loss_weight = args.d_loss_weight
        self.edge_dir = args.edge_dir
        self.test_result_dir = args.test_result_dir


    def generator(self, input, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            resNum = 7
            # conv
            x = conv(input, channels=32, kernel=3, stride=1, scope='convlayer_1',is_training=is_training)
            x = batch_norm(x, is_training=is_training, scope='conv_batch_norm_1')
            x = relu(x)
            x = conv(x, channels=32, kernel=3, stride=1, scope='convlayer_2', is_training=True)
            x = batch_norm(x, is_training=is_training, scope='conv_batch_norm_2')
            x = relu(x)
            x = conv(x, channels=32, kernel=3, stride=2, scope='convlayer_3', is_training=True)
            x = batch_norm(x, is_training=is_training, scope='conv_batch_norm_3')
            x = relu(x)
            x = conv(x, channels=32, kernel=3, stride=2, scope='convlayer_4', is_training=True)
            x = batch_norm(x, is_training=is_training, scope='conv_batch_norm_4')
            x = relu(x)
            # resblock
            for i in range(resNum):
                x = resblock(x, channels=32, is_training=is_training, use_bias=True, downsample=False,
                             scope='resblock_' + str(i))
            # deconv
            x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=4, strides=2, padding='SAME', )
            x = batch_norm(x, is_training=is_training, scope='conv_batch_norm_30')
            x = relu(x)
            x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=4, strides=2, padding='SAME', trainable=is_training)
            x = batch_norm(x, is_training=is_training, scope='conv_batch_norm_31')
            x = relu(x)
            # conv
            x = conv(x, channels=32, kernel=3, stride=1, scope='convlayer_32')
            x = batch_norm(x, is_training=is_training, scope='conv_batch_norm_32')
            x = relu(x)
            x = conv(x, channels=1, kernel=3, stride=1, scope='convlayer_33', is_training=is_training)
            return x

    def sn_discriminator(self, x, is_training = True, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            cnum = 32
            x = gen_snconv(x, cnum, 5, 2, name='dis_conv1', training=is_training)
            x = gen_snconv(x, cnum*2, 5, 2, name='dis_conv2', training=is_training)
            x = gen_snconv(x, cnum*4, 5, 2, name='dis_conv3', training=is_training)
            x = gen_snconv(x, 1, 5, 1, name='dis_conv4', training=is_training)
        return x

    def build_model(self):
        self.input = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.output = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.mask = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.edge = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        """ Loss Function"""
        # discriminator loss
        self.mask_down = downsample(self.mask,8)
        D_real_logit = self.sn_discriminator(self.input - self.output)
        G = self.generator(self.input, self.edge)
        D_fake_logit = self.sn_discriminator(self.input - G, reuse=True)
        self.d_loss_fake = tf.reduce_mean(tf.square(tf.sigmoid(D_fake_logit)))
        self.d_loss_real = tf.reduce_mean(tf.multiply(self.mask_down,tf.square(tf.sigmoid(D_real_logit)-1)))
        self.d_loss = self.d_loss_fake + self.d_loss_real
        # genenrator loss
        self.g_loss = tf.reduce_mean(tf.multiply(self.mask, tf.abs(tf.subtract(self.output,G))))\
            + 2*tf.reduce_mean(tf.multiply(self.mask, tf.abs(tf.subtract(grad_x(self.output),grad_x(G)))))\
            + 2*tf.reduce_mean(tf.multiply(self.mask, tf.abs(tf.subtract(grad_y(self.output),grad_y(G)))))\
            + 0.1*tf.reduce_mean(tf.square(tf.sigmoid(D_fake_logit)-1))
        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        # optimizers
        self.d_optim = tf.train.AdamOptimizer(0.01).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.01).minimize(self.g_loss, var_list=g_vars)

        """ Testing """
        self.fake_bone = self.generator(self.input, self.edge, is_training=False, reuse=True)
        self.d_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)

    
    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()


        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.train_writer = tf.summary.FileWriter(self.log_dir+'/train', self.sess.graph)
        #self.val_writer = tf.summary.FileWriter(self.log_dir+'/test')

        # restore check-point if it exits
        #could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(self.checkpoints_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoints_dir,ckpt_name))
            start_epoch = int(next(re.finditer("(\d+)", ckpt_name)).group()) + 1
            counter = start_epoch*self.num_batches + 1
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            counter = 1
            print(" [!] Load failed...")
        
        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.num_epochs):
            # get batch data
            id_list = np.random.permutation(self.train_num)
            for i in range(0, self.num_batches):
                input_image_batch = []
                output_image_batch = []
                mask_image_batch = []
                edge_image_batch = []
                for j in range(self.batch_sz):
                    idx = i * self.batch_sz + j
                    id = id_list[idx]
                    train_img = sio.loadmat(self.train_files[id])
                    input_image = np.float32(train_img['I'])
                    output_image = np.float32(train_img['B'])
                    mask_image = np.float32(1 - train_img['invalidMask'])
                    patientname = filepath_to_name(self.train_files[id])
                    edgepath = self.edge_dir + patientname + '.mat'
                    edge_image = sio.loadmat(edgepath)
                    edge_image = edge_image['edge']
                    input_image, output_image, mask_image, edge_image = random_crop(input_image, output_image, mask_image, edge_image, self.crop_h, self.crop_w)
                    input_image_batch.append(input_image)
                    output_image_batch.append(output_image)
                    mask_image_batch.append(mask_image)
                    edge_image_batch.append(edge_image)
                if self.batch_sz == 1:
                    input_image_batch = input_image_batch[0]
                    output_image_batch = output_image_batch[0]
                    mask_image_batch = mask_image_batch[0]
                    edge_image_batch = edge_image_batch[0]
                    input_image_batch = np.expand_dims(np.expand_dims(input_image_batch, axis=0), axis=3)
                    output_image_batch = np.expand_dims(np.expand_dims(output_image_batch, axis=0), axis=3)
                    mask_image_batch = np.expand_dims(np.expand_dims(mask_image_batch, axis=0), axis=3)
                    edge_image_batch = np.expand_dims(np.expand_dims(edge_image_batch, axis=0), axis=3)
                else:
                    input_image_batch = np.stack(input_image_batch, axis=0)
                    output_image_batch = np.stack(output_image_batch, axis=0)
                    mask_image_batch = np.stack(mask_image_batch, axis=0)
                    edge_image_batch = np.stack(edge_image_batch, axis=0)
                    input_image_batch = np.expand_dims(input_image_batch, axis=3)
                    output_image_batch = np.expand_dims(output_image_batch, axis=3)
                    mask_image_batch = np.expand_dims(mask_image_batch, axis=3)
                    edge_image_batch = np.expand_dims(edge_image_batch, axis=3)


                train_feed_dict = {
                    self.input : input_image_batch,
                    self.output : output_image_batch,
                    self.edge : edge_image_batch,
                    self.mask : mask_image_batch
                }


                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict=train_feed_dict)
                self.train_writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict=train_feed_dict)
                self.train_writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.4f, g_loss: %.4f" \
                      % (epoch, i, self.num_batches, time.time() - start_time, d_loss, g_loss))

            # save model
            self.saver.save(self.sess, "%s/%d-model.ckpt" % (self.checkpoints_dir, epoch))

            # save results
            if epoch % 10 == 0:
                for k in range(self.val_num):
                    val_cxr = sio.loadmat(self.val_files[k])
                    val_cxr = np.float32(val_cxr['I'])
                    file_name = filepath_to_name(self.val_files[k])
                    edge_dir = self.edge_dir + file_name + '.mat'
                    val_edge = sio.loadmat(edge_dir)
                    val_edge = val_edge['edge']
                    val_input_image = np.expand_dims(np.expand_dims(val_cxr, axis=0), axis=3)
                    val_edge_image = np.expand_dims(np.expand_dims(val_edge, axis=0), axis=3)
                    predicted = self.sess.run(self.fake_bone, feed_dict={self.input: val_input_image, self.edge: val_edge_image})
                    predicted = np.squeeze(predicted)
                    sio.savemat("%s/%04d_%s_predicted.mat" % (self.result_dir, epoch, file_name),
                                {'predicted': predicted})

    def test(self, input, edge, file_name):
        # saver to save model
        self.saver = tf.train.Saver()

        # restore check-point if it exits
        ckpt = tf.train.get_checkpoint_state(self.checkpoints_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoints_dir,ckpt_name))
            print(" [*] Load SUCCESS")
        predicted = self.sess.run(self.fake_bone, feed_dict={self.input: input, self.edge: edge})
        predicted = np.squeeze(predicted)
        sio.savemat("%s/%s_predicted.mat" % (self.test_result_dir, file_name),
                                {'predicted': predicted})
        


