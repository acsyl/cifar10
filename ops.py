import tensorflow as tf
import numpy as np

he_normal = tf.contrib.keras.initializers.he_normal()

def batch_norm_layer(inputs, is_training, momentum=0.9, epsilon=1e-5, in_place_update=True, name="batch_norm"):
    '''
    Helper function to create a batch normalization layer
    '''
    if in_place_update:
        return tf.contrib.layers.batch_norm(inputs, decay=momentum, epsilon=epsilon,
                                            center=True, scale=True, updates_collections=None,
                                            is_training=is_training, scope=name)
    else:
        return tf.contrib.layers.batch_norm(inputs, decay=momentum, epsilon=epsilon,
                                            center=True, scale=True, is_training=is_training, scope=name)

def Conv(inputs, num_filters, is_training, name):
    # Conv2D + Batch norm + ReLU layers
    with tf.variable_scope("conv_block_%s" % name):
        filter_shape = [3, 3, inputs.get_shape()[3], num_filters]
        w = tf.get_variable(name='W_1', shape=filter_shape, 
            initializer=he_normal)
        b = tf.get_variable(name='b_1', shape=[num_filters], 
                initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="SAME") + b
    return conv


