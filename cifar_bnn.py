import tensorflow as tf
import numpy as np
import time
import os.path

import bnn
#import tf_export

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def cifar10():
    batch = []
    labels = []
    size = 32*32*3+1
    for path in ['data_batch_%i' % (i + 1) for i in range(5)] + ['test_batch']:
        d = open('data/cifar-10-batches-bin/' + path + '.bin', 'rb').read()
        assert(len(d) % size == 0)
        for i in range(0, len(d), size):
            e = d[i:i+size]
            labels += [e[0]]
            batch += [np.frombuffer(e[1:], dtype=np.uint8)]

    data = np.concatenate(batch)
    data = data.astype(np.float32)
    data = np.multiply(data, 2.0 / 255.0)
    data = np.add(data, -1.0)

    data = np.reshape(data, (-1, 3, 32, 32))
    data = np.transpose(data, (0, 2, 3, 1))
    data = np.reshape(data, (-1, 32*32*3))

    label = dense_to_one_hot(np.asarray(labels), 10)
    return data[:50000], label[:50000], data[50000:], label[50000:]


train_x, train_y, test_x, test_y = cifar10()

x0 = tf.placeholder(tf.float32, [None, 32*32*3])
y0 = tf.placeholder(tf.float32, [None, 10])
train = tf.placeholder(tf.bool, name='is_training')
lr = tf.placeholder(tf.float32)

# convolutions
x = tf.reshape(x0, [-1, 32, 32, 3])
x = bnn.layer(x, 128, filter_size=[3, 3])
x = bnn.layer(x, 128, filter_size=[3, 3], pool=([2, 2], [2, 2]))
x = bnn.layer(x, 256, filter_size=[3, 3])
x = bnn.layer(x, 256, filter_size=[3, 3], pool=([2, 2], [2, 2]))
x = bnn.layer(x, 512, filter_size=[3, 3])
x = bnn.layer(x, 512, filter_size=[3, 3], pool=([2, 2], [2, 2]))

# fully connected
x = bnn.layer(x, 1024, filter_size=[4, 4], padding='VALID')
x = bnn.layer(x, 1024)
x = bnn.layer(x, 10, activate='none')
_y = tf.identity(x)
y = tf.reshape(_y, [-1, 10])

loss = tf.reduce_mean(tf.square(tf.losses.hinge_loss(y0, y)))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
update_op = tf.group(*update_ops)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y0,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    if os.path.isfile("modelcifarx.ckpt.meta"):
        print('loading model')
        saver.restore(sess, "modelcifarx.ckpt")

    #bnn.export(y, x0, (1, 3072), 2.0 / 255.0, -1.0, 'cifar')

    #yy = sess.run([y], feed_dict={x0: test_x[0:1], y0: test_y[0:1], train : False})
    #print(yy)

    print('training...')

#Training
    EPOCHS = 500
    BATCH_SIZE = 50
    LR = 0.001
    LR_DECAY = (0.0000003/LR)**(1.0/EPOCHS)

    print(LR_DECAY)

    num_batch = len(train_x) / BATCH_SIZE

    from sklearn.utils import shuffle

    for i in range(EPOCHS):
        tx, ty = shuffle(train_x, train_y)
        total_loss = 0.0

        t0 = time.perf_counter()

        for off in range(0, len(train_x), BATCH_SIZE):
            end = off + BATCH_SIZE
            x, y = tx[off:end], ty[off:end]

            _, l = sess.run([train_step, loss], feed_dict={x0: x, y0: y, train : True, lr : LR})
            total_loss += l

            sess.run(update_op)

        t1 = time.perf_counter()

        total_loss /= num_batch
        LR *= LR_DECAY

        # split in batches of 100 because memory
        ac2 = 0.0
        l2 = 0.0
        for j in range(0, 10000, 100):
            ac, l = sess.run([accuracy, loss], feed_dict={x0: test_x[j:j+100], y0: test_y[j:j+100], train : False})
            ac2 += ac
            l2 += l
        ac2 /= 100.0
        l2 /= 100.0
        print("epoch %i: accuracy=%f,%f loss=%f time=%f" % (i, ac2, l2, total_loss, t1 - t0))
        save_path = saver.save(sess, "modelcifarx.ckpt")

