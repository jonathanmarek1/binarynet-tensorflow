import tensorflow as tf
import bnn
import tf_export

x0 = tf.placeholder(tf.float32, [None, 1, 1, 4096])
x1 = bnn.activation(x0)

y0 = bnn.layer(x0, 4096, activate='none', norm=False, binary=False)
y1 = bnn.layer(x0, 4096, activate='none', norm=False)
y2 = bnn.layer(x1, 4096, activate='none', norm=False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    tf_export.export(y0, x0, 'benchmark_float', False)
    tf_export.export(y0, x0, 'benchmark_int8', True)
    tf_export.export(y1, x0, 'benchmark_float_bin', False)
    tf_export.export(y1, x0, 'benchmark_int8_bin', True)
    tf_export.export(y2, x0, 'benchmark_bin', False)
