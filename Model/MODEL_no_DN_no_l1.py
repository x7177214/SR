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
        neg = alphas * (x - abs(x)) * 0.5

    return pos + neg, weights

def model(input_tensor, bic_tesor, r, train):
    weights = []
    loss_v_l1 = []
    tensor = None

    # Conv1 5x5@32
    conv_1_w = tf.get_variable("conv_1_w", [5, 5, 1, 32],
                               initializer=tf.random_normal_initializer(stddev=0.001))
    conv_1_b = tf.get_variable("conv_1_b", [32],
                               initializer=tf.constant_initializer(0))
    weights.append(conv_1_w)
    weights.append(conv_1_b)
    tensor = tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_1_w,
                                         strides=[1, 1, 1, 1], padding='SAME'), conv_1_b)
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
    if train:
        # Sub-pixel layer
        tensor = sb(tensor, r)

        # Residual
        tensor = tf.add(tensor, bic_tesor)

    return tensor, weights
