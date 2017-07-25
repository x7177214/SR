import tensorflow as tf
import numpy as np
from Subpixel import sb



A = tf.placeholder(tf.float32, shape=(1, 5, 5, 1))

dim = A.get_shape().as_list()
Aa = tf.nn.avg_pool(A, ksize=[1, dim[1], dim[2], 1],
                              strides=[1, 1, 1, 1], padding='VALID')


# a = np.random(25).reshape([1, 5, 5, 1])
a = np.random.rand(1, 5, 5, 1)
aaa = np.reshape(a, [5, 5])

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    aa = sess.run([Aa], feed_dict={A: a})



    # aa = np.reshape(aa, [1, 1, 1])

    print aa.shape
    # print aa