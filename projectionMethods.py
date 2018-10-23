from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorlayer.layers import *

NUM_CLASSES = 10
G_PROJECTIONS = [16,16,16]

def batch_norm_map(x,classes,v, i):
    '''manual batch norm because tensorflow can't set beta or gamma'''
    with tf.variable_scope('batch_norm_%i'%(i)) as scope:
        class_max =tf.argmax(classes, axis = 1)
        m = tf.shape(class_max)
        m = tf.concat([m,[G_PROJECTIONS[i]]],0)
        z = tf.random_normal(m)
        epsilon = 1e-8
        mean, variance = tf.nn.moments(x, axes=[0,1,2])
        x_normed = (x - mean) / tf.sqrt(variance + epsilon)
        class_matrix = tf.get_variable('gen_mapping_w_%i'%(i), [NUM_CLASSES, G_PROJECTIONS[i]])
        bias_matrix = tf.get_variable('gen_mapping_b_%i'%(i), [NUM_CLASSES, G_PROJECTIONS[i]])
        class_selection = tf.expand_dims(class_max, -1)
        selected_weights = tf.gather_nd(class_matrix, class_selection)
        selected_biases =  tf.gather_nd(bias_matrix, class_selection)
        z2 = z * selected_weights + selected_biases
        z2 = tf.concat([z2,classes], axis = -1)
        batch_gamma = tf.layers.dense(z2, abs(v), bias_initializer = tf.ones_initializer(), name='gamma_projection_%i'%(i))
        batch_beta = tf.layers.dense(z2, abs(v), bias_initializer = tf.zeros_initializer(), name='beta_projection_%i'%(i))
        batch_beta = tf.expand_dims(tf.expand_dims(batch_beta,1),1)
        _,xs,ys,_ = x_normed.get_shape()
        batch_beta = tf.tile(batch_beta,[1,xs, ys,1])
        new_x = tf.einsum('abcd,ad->abcd',x_normed, batch_gamma) + batch_beta
        return new_x

def batch_norm_disc(x,classes,v, i):
    '''manual batch norm because tensorflow can't set beta or gamma'''
    with tf.variable_scope('batch_norm_%i'%(i)) as scope:
        print(classes.get_shape())
        class_max =tf.argmax(classes, axis = 1)
        print(class_max.get_shape())
        epsilon = 1e-8
        mean, variance = tf.nn.moments(x, axes=[0,1,2])
        x_normed = (x - mean) / tf.sqrt(variance + epsilon)
        class_matrix = tf.get_variable('d_mapping_w_%i'%(i), [NUM_CLASSES, abs(v)])
        bias_matrix = tf.get_variable('d_mapping_b_%i'%(i), [NUM_CLASSES, abs(v)])
        class_selection = tf.expand_dims(class_max, -1)
        batch_gamma = tf.gather_nd(class_matrix, class_selection)
        batch_beta =  tf.gather_nd(bias_matrix, class_selection)
        batch_beta = tf.expand_dims(tf.expand_dims(batch_beta,1),1)
        _,xs,ys,_ = x_normed.get_shape()
        batch_beta = tf.tile(batch_beta,[1,xs, ys,1])
        new_x = tf.einsum('abcd,ad->abcd',x_normed, batch_gamma) + batch_beta
        return new_x

def batch_norm_sota(x,classes,v, i):
    '''manual batch norm because tensorflow can't set beta or gamma'''
    with tf.variable_scope('batch_norm_%i'%(i)) as scope:
        epsilon = 1e-8
        mean, variance = tf.nn.moments(x, axes=[0,1,2])
        z = tf.random_normal(tf.shape(classes))
        x_normed = (x - mean) / tf.sqrt(variance + epsilon)
        z2 = tf.concat([z, classes],-1)
        batch_gamma = tf.layers.dense(z2, abs(v), bias_initializer = tf.ones_initializer(), name='gamma_projection_%i'%(i))
        batch_beta = tf.layers.dense(z2, abs(v), bias_initializer = tf.zeros_initializer(), name='beta_projection_%i'%(i))
        batch_beta = tf.expand_dims(tf.expand_dims(batch_beta,1),1)
        _,xs,ys,_ = x_normed.get_shape()
        batch_beta = tf.tile(batch_beta,[1,xs, ys,1])
        new_x = tf.einsum('abcd,ad->abcd',x_normed, batch_gamma) + batch_beta
        return new_x

def feature_concat(x,classes):
    classes = tf.expand_dims(tf.expand_dims(classes,1),1)
    _,xs,ys,_ = x.get_shape()
    classes = tf.tile(classes,[1,xs, ys,1])
    x = tf.concat([x,classes], axis = -1)
    return x
