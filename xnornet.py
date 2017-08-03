"""
XNORNet / BWN implementation, importing pretrained weights with pytorch
"""

import tensorflow as tf
import numpy as np

import torch
import torch.legacy.nn as nn
nn.BinActiveZ = nn.ReLU # so the deserialization doesnt fail
from torch.utils.serialization import load_lua

import bnn
import tf_export

BWN = True

cache = load_lua('data/cache/meanstdCache.t7')
model = load_lua('data/alexnet_BWN.t7' if BWN else 'data/alexnet_XNOR.t7')

x0 = tf.placeholder(tf.float32, [None, 227, 227, 3])
train = tf.placeholder(tf.bool, name='is_training')

x = (x0 * (1.0 / 255.0) - np.array(cache.mean)) * (1.0 / np.array(cache.std))

x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], 'CONSTANT')

if BWN:
    x, x1 = bnn.layer2(x, 96, filter_size=[11, 11], stride=[4, 4], pool=([3, 3], [2, 2]), epsilon=0.001, binary=False, padding='VALID', activate='relu')

else:
    # TODO
    x = bnn.layer_xnornet0(x, 96, filter_size=[11, 11], stride=[4, 4], pool=([3, 3], [2, 2]), epsilon=0.00001, binary=False, padding='VALID')

if BWN:
    act = 'relu'
    eps = 0.001
else:
    act = 'bin'
    eps = 0.0001

x = bnn.layer(x, 256, filter_size=[5, 5], pool=([3, 3], [2, 2]), activate=act, epsilon=eps)

x = bnn.layer(x, 384, filter_size=[3, 3], activate=act, epsilon=eps)
x = bnn.layer(x, 384, filter_size=[3, 3], activate=act, epsilon=eps)
x = bnn.layer(x, 256, filter_size=[3, 3], pool=([3, 3], [2, 2]), activate=act, epsilon=eps)

x = bnn.layer(x, 4096, filter_size=[6, 6], padding='VALID', activate=act, epsilon=eps)
x = bnn.layer(x, 4096, activate='relu', epsilon=0.001)
x = bnn.layer(x, 1000, activate='none', norm=False, binary=False)

y = tf.identity(x)

softmax = tf.nn.softmax(y)

# helper function to load torch batch norm weights into tensorflow batch norm variables
def load_batch_norm(scope, module):
    beta = gamma = mean = variance = None
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/'):
        if i.name == scope+'/beta:0':
            assert(beta == None)
            beta = i
        elif i.name == scope+'/gamma:0':
            assert(gamma == None)
            gamma = i
        elif i.name == scope+'/moving_mean:0':
            assert(mean == None)
            mean = i
        elif i.name == scope+'/moving_variance:0':
            assert(variance == None)
            variance = i
        else:
            assert(False)

    assert(beta is not None and gamma is not None)
    assert(mean is not None and variance is not None)

    sess.run(tf.group(tf.assign(beta, module.bias.numpy()),
                      tf.assign(gamma, module.weight.numpy()),
                      tf.assign(mean, module.running_mean.numpy()),
                      tf.assign(variance, module.running_var.numpy())))

# helper function to load convolution weights into tensorflow variables
def load_conv_param(w, b, module):
    w = [i for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if i.name == w][0]
    b = [i for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if i.name == b][0]
    W = module.weight.numpy()
    W = np.transpose(W, (3, 2, 1, 0))

    sess.run(tf.group(tf.assign(w, W), tf.assign(b, module.bias.numpy())))

from scipy.misc import imread

img = (imread('../summer/xnor/laska.png')[:,:,:3]).astype(np.float32)

img = np.transpose(img, (1, 0, 2))
img = img[np.newaxis]
print(img.shape)

with tf.Session() as sess:
    if BWN:
        load_batch_norm('BatchNorm', model.modules[0].modules[1])
        load_batch_norm('BatchNorm_1', model.modules[2].modules[1])
        load_batch_norm('BatchNorm_2', model.modules[4].modules[1])
        load_batch_norm('BatchNorm_3', model.modules[5].modules[1])
        load_batch_norm('BatchNorm_4', model.modules[6].modules[1])
        load_batch_norm('BatchNorm_5', model.modules[9].modules[1])
        load_batch_norm('BatchNorm_6', model.modules[11].modules[1])

        load_conv_param('Variable:0', 'Variable_1:0', model.modules[0].modules[0])
        load_conv_param('Variable_2:0', 'Variable_4:0', model.modules[2].modules[0])
        load_conv_param('Variable_5:0', 'Variable_7:0', model.modules[4].modules[0])
        load_conv_param('Variable_8:0', 'Variable_10:0', model.modules[5].modules[0])
        load_conv_param('Variable_11:0', 'Variable_13:0', model.modules[6].modules[0])
        load_conv_param('Variable_14:0', 'Variable_16:0', model.modules[9].modules[0])
        load_conv_param('Variable_17:0', 'Variable_19:0', model.modules[11].modules[0])
        load_conv_param('Variable_20:0', 'Variable_21:0', model.modules[12])
    else:
        load_batch_norm('BatchNorm', model.modules[1])
        load_batch_norm('BatchNorm_1', model.modules[4].modules[0])
        load_batch_norm('BatchNorm_2', model.modules[5].modules[0])
        load_batch_norm('BatchNorm_3', model.modules[6].modules[0])
        load_batch_norm('BatchNorm_4', model.modules[7].modules[0])
        load_batch_norm('BatchNorm_5', model.modules[8].modules[0])
        load_batch_norm('BatchNorm_6', model.modules[9].modules[0])
        load_batch_norm('BatchNorm_7', model.modules[10])

        load_conv_param('Variable:0', 'Variable_1:0', model.modules[0])
        load_conv_param('Variable_2:0', 'Variable_4:0', model.modules[4].modules[2])
        load_conv_param('Variable_5:0', 'Variable_7:0', model.modules[5].modules[2])
        load_conv_param('Variable_8:0', 'Variable_10:0', model.modules[6].modules[2])
        load_conv_param('Variable_11:0', 'Variable_13:0', model.modules[7].modules[2])
        load_conv_param('Variable_14:0', 'Variable_16:0', model.modules[8].modules[2])
        load_conv_param('Variable_17:0', 'Variable_19:0', model.modules[9].modules[2])
        load_conv_param('Variable_20:0', 'Variable_21:0', model.modules[12])

    output, x1 = sess.run([softmax, x1], feed_dict={x0 : img, train: False})

    print(x1[0,0,0,0], np.max(x1))

    output = output[0,0,0,:]

    order = sorted(range(len(output)), key=lambda x: output[x], reverse=True)
    for i in range(5):
        print(order[i], output[order[i]])

    tf_export.export(y, x0, 'xnornet_bwn' if BWN else 'xnornet', True)
