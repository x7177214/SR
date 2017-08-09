import tensorflow as tf
import numpy as np
from Subpixel import sb

def prelu(x, weights, scope='prelu'):
    with tf.variable_scope(scope):
        alphas = tf.get_variable('alpha', x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        weights.append(alphas)
        pos = tf.nn.relu(x)
        neg = alphas * 0.5 * (x - abs(x)) 

    return pos + neg, weights

def div_norm(input_, weights, loss_v_l1, name="div_norm", k_size=3):
    with tf.variable_scope(name):
        mean = tf.nn.avg_pool(input_, ksize=[1, k_size, k_size, 1],
                              strides=[1, 1, 1, 1], padding='SAME')
        mean = tf.reduce_mean(mean, axis=3, keep_dims=True)
        v = input_ - mean
        var = tf.nn.avg_pool(v ** 2, ksize=[1, k_size, k_size, 1],
                              strides=[1, 1, 1, 1], padding="SAME")
        var = tf.reduce_mean(var, axis=3, keep_dims=True)
        sigma = tf.get_variable('sigma', [1],
                                initializer=tf.constant_initializer(0.0))
        weights.append(sigma)
        loss_v_l1.append(tf.reduce_mean(tf.sqrt(v**2 + (1e-3)**2)))

        return v / tf.sqrt(var + sigma**2), weights, loss_v_l1

def model(input_tensor, r):
    weights = []
    loss_v_l1 = []
    tesor = None

    # Conv1 5x5@32
    conv_1_w = tf.get_variable("conv_1_w", [5, 5, 1, 32],
                               initializer=tf.random_normal_initializer(stddev=0.001))
    conv_1_b = tf.get_variable("conv_1_b", [32],
                               initializer=tf.constant_initializer(0))
    weights.append(conv_1_w)
    weights.append(conv_1_b)
    tensor = tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_1_w,
                                         strides=[1, 1, 1, 1], padding='SAME'), conv_1_b)
    tensor, weights, loss_v_l1 = div_norm(tensor, weights, loss_v_l1, name='norm_1')
    tensor, weights = prelu(tensor, weights, scope='prelu_1')

    # Conv2 1x1@5
    conv_2_w = tf.get_variable("conv_2_w", [1, 1, 32, 5],
                               initializer=tf.random_normal_initializer(stddev=0.001))
    conv_2_b = tf.get_variable("conv_2_b", [5],
                               initializer=tf.constant_initializer(0))
    weights.append(conv_2_w)
    weights.append(conv_2_b)
    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_2_w,
                                         strides=[1, 1, 1, 1], padding='SAME'), conv_2_b)
    tensor, weights, loss_v_l1 = div_norm(tensor, weights, loss_v_l1, name='norm_2')
    tensor, weights = prelu(tensor, weights, scope='prelu_2')

    # Conv3 3x3@5
    conv_3_w = tf.get_variable("conv_3_w", [3, 3, 5, 5],
                               initializer=tf.random_normal_initializer(stddev=0.001))
    conv_3_b = tf.get_variable("conv_3_b", [5],
                               initializer=tf.constant_initializer(0))
    weights.append(conv_3_w)
    weights.append(conv_3_b)
    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_3_w,
                                         strides=[1, 1, 1, 1], padding='SAME'), conv_3_b)
    tensor, weights, loss_v_l1 = div_norm(tensor, weights, loss_v_l1, name='norm_3')
    tensor, weights = prelu(tensor, weights, scope='prelu_3')

    # Conv4 3x3@32
    conv_4_w = tf.get_variable("conv_4_w", [3, 3, 5, 32],
                               initializer=tf.random_normal_initializer(stddev=0.001))
    conv_4_b = tf.get_variable("conv_4_b", [32],
                               initializer=tf.constant_initializer(0))
    weights.append(conv_4_w)
    weights.append(conv_4_b)
    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_4_w,
                                         strides=[1, 1, 1, 1], padding='SAME'), conv_4_b)
    tensor, weights, loss_v_l1 = div_norm(tensor, weights, loss_v_l1, name='norm_4')
    tensor, weights = prelu(tensor, weights, scope='prelu_4')

    # Conv5 3x3@r^2
    conv_5_w = tf.get_variable("conv_5_w", [3, 3, 32, r * r],
                               initializer=tf.random_normal_initializer(stddev=0.001))
    conv_5_b = tf.get_variable("conv_5_b", [r * r],
                               initializer=tf.constant_initializer(0))
    weights.append(conv_5_w)
    weights.append(conv_5_b)
    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_5_w,
                                         strides=[1, 1, 1, 1], padding='SAME'), conv_5_b)
    
    #################### res_tesor #####################
    # Sub-pixel layer(res)
    res_tesor = sb(tensor, r)
    ####################################################

    #################### main_tesor ####################
    # input(1) -> 3x3 -> ouput(r * r)

    template = np.random.rand(3, 3, 1, 1) * 0.001
    template = np.array(template, dtype='float32')
    template[1, 1, :, :] = 1.
    init = tf.constant(template)
    conv_0a_w = tf.get_variable("conv_0a_w", initializer=init, dtype=tf.float32)
    
    template = np.random.rand(3, 3, 1, 1) * 0.001
    template = np.array(template, dtype='float32')
    template[1, 1, :, :] = 0.5
    template[1, 2, :, :] = 0.5
    init = tf.constant(template)
    conv_0b_w = tf.get_variable("conv_0b_w", initializer=init, dtype=tf.float32)

    template = np.random.rand(3, 3, 1, 1) * 0.001
    template = np.array(template, dtype='float32')
    template[1, 1, :, :] = 0.5
    template[2, 1, :, :] = 0.5
    init = tf.constant(template)
    conv_0c_w = tf.get_variable("conv_0c_w", initializer=init, dtype=tf.float32)

    template = np.random.rand(3, 3, 1, 1) * 0.001
    template = np.array(template, dtype='float32')
    template[0, 0, :, :] = 0.25
    template[2, 0, :, :] = 0.25
    template[0, 2, :, :] = 0.25
    template[2, 2, :, :] = 0.25
    init = tf.constant(template)
    conv_0d_w = tf.get_variable("conv_0d_w", initializer=init, dtype=tf.float32)
    
    conv_0_b = tf.get_variable("conv_0_b", [r * r], initializer=tf.constant_initializer(0))
    weights.append(conv_0a_w)
    weights.append(conv_0b_w)
    weights.append(conv_0c_w)
    weights.append(conv_0d_w)
    weights.append(conv_0_b)

    fa = tf.nn.conv2d(input_tensor, conv_0a_w, strides=[1, 1, 1, 1], padding='SAME')
    fb = tf.nn.conv2d(input_tensor, conv_0b_w, strides=[1, 1, 1, 1], padding='SAME')
    fc = tf.nn.conv2d(input_tensor, conv_0c_w, strides=[1, 1, 1, 1], padding='SAME')
    fd = tf.nn.conv2d(input_tensor, conv_0d_w, strides=[1, 1, 1, 1], padding='SAME')

    concat_tensor = tf.concat([fa, fb, fc, fd], 3)
    main_tesor = tf.nn.bias_add(concat_tensor, conv_0_b)

    # Sub-pixel layer(main)
    main_tesor = sb(main_tesor, r)
    ####################################################

    tensor = tf.add(res_tesor, main_tesor)

    return tensor, weights, loss_v_l1
