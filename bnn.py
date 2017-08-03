import tensorflow as tf
import numpy as np

# code

# binary activation function
def activation(x):
    x = tf.clip_by_value(x, -1.0, 1.0)
    return x + tf.stop_gradient(tf.sign(x) - x)

# create weight + bias variables with update op as in BinaryNet
def weight_bias(shape, binary=True, init=None):
    print(shape)
    init = tf.random_uniform(shape, -1.0, 1.0) if init is None else init
    print(init)
    x = tf.Variable(init)

    if binary:
        y = tf.Variable(init)

        coeff = np.float32(1./np.sqrt(1.5/ (np.prod(shape[:-2]) * (shape[-2] + shape[-1]))))
        print(coeff)

        tmp = y + coeff * (x - y)
        tmp = tf.clip_by_value(tmp, -1.0, 1.0)
        tmp = tf.group(x.assign(tmp), y.assign(tmp))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tmp)

        x = tf.clip_by_value(x, -1.0, 1.0)
        xbin = tf.sign(x) * tf.reduce_mean(tf.abs(x), axis=[0, 1, 2])
        x = x + tf.stop_gradient(xbin - x)

    return x, tf.Variable(tf.constant(0.1, shape=[shape[-1]]))

def batch_norm(x, epsilon, decay=0.9):
    train = tf.get_default_graph().get_tensor_by_name('is_training:0')
    return tf.contrib.layers.batch_norm(x, decay=decay, center=True, scale=True,
        epsilon=epsilon, updates_collections=None, is_training=train, trainable=True,
        fused=True)

# a layer in BinaryNet
def layer(x, num_output, filter_size=[1, 1], stride=[1, 1], pool=None, activate='bin',
          binary=True, norm=True, epsilon=0.0001, padding='SAME', init=None):
    shape = filter_size + [x.shape[-1].value, num_output]

    W, b = weight_bias(shape, binary, init)

    x = tf.nn.conv2d(x, W, strides=[1, *stride, 1], padding=padding) + b

    if norm:
        x = batch_norm(x, epsilon)

    if pool is not None:
        x = tf.nn.max_pool(x, ksize=[1, *pool[0], 1], strides=[1, *pool[-1], 1], padding='VALID')

    if activate == 'bin':
        return bnn.active(x)
    elif activate == 'relu':
        return tf.nn.relu(x)

    assert(activate == 'none')
    return x

def layer2(x, num_output, filter_size=[1, 1], stride=[1, 1], pool=None, activate='bin',
          binary=True, norm=True, epsilon=0.0001, padding='SAME', init=None):
    shape = filter_size + [x.shape[-1].value, num_output]

    W, b = weight_bias(shape, binary, init)

    x = tf.nn.conv2d(x, W, strides=[1, *stride, 1], padding=padding) + b

    if norm:
        x = batch_norm(x, epsilon)

    if pool is not None:
        x = tf.nn.max_pool(x, ksize=[1, *pool[0], 1], strides=[1, *pool[-1], 1], padding='VALID')

    if activate == 'bin':
        return bnn.active(x)
    elif activate == 'relu':
        return (tf.nn.relu(x), x)

    assert(activate == 'none')
    return x

def layer_xnornet0(x, num_output, filter_size=[1, 1], stride=[1, 1], pool=None, activate=True, binary=True, norm=True, epsilon=0.0001, padding='SAME', init=None):
    shape = filter_size + [x.shape[-1].value, num_output]

    W, b = weight_bias(shape, binary, init)

    x = tf.nn.conv2d(x, W, strides=[1, *stride, 1], padding=padding) + b

    x = batch_norm(x, epsilon)
    x = tf.nn.relu(x)

    if pool is not None:
        x = tf.nn.max_pool(x, ksize=[1, *pool[0], 1], strides=[1, *pool[-1], 1], padding='VALID')

    if norm:
        x = batch_norm(x, 0.0001)

    return activation(x) if activate else x

