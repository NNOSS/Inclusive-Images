from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorlayer.layers import *

NUM_CLASSES = 20

def return_rep(WHICH_MODEL,x,classes,v, z, embedding, i, i2):
    if WHICH_MODEL == 0:
        batch = BatchNormLayer(x,act=tf.nn.relu, is_train=True,name='gen_bn_%i_%i'%(i,i2))
        batch = feature_concat(batch.outputs, classes)
    elif WHICH_MODEL ==1:
        batch = batch_norm_sota(x.outputs,classes, v, z,embedding, i,i2)
    elif WHICH_MODEL == 2:
        batch = batch_norm_map(x.outputs,classes, v, z,embedding, i,i2)
    elif WHICH_MODEL == 3:
        batch = batch_norm_cond(x.outputs,classes, v, embedding, i,i2)
    return batch

def batch_norm_map(x,classes,v, z, embedding,i,i2):
    '''manual batch norm because tensorflow can't set beta or gamma'''
    with tf.variable_scope('batch_norm_%i_%i'%(i,i2)) as scope:
        class_max =tf.argmax(classes, axis = 1)
        # m = tf.shape(class_max)
        # m = tf.concat([m,[G_PROJECTIONS[i]]],0)
        # z = tf.random_normal(m)
        epsilon = 1e-4
        mean, variance = tf.nn.moments(x, axes=[0,1,2])
        x_normed = (x - mean) / tf.sqrt(variance + epsilon)
        z_size = z.get_shape()[1]
        # class_matrix = tf.get_variable('gen_mapping_w_%i_%i'%(i,i2), [NUM_CLASSES, z_size])
        # bias_matrix = tf.get_variable('gen_mapping_b_%i_%i'%(i,i2), [NUM_CLASSES, z_size])
        # class_selection = tf.expand_dims(class_max, -1)
        # selected_weights = tf.gather_nd(class_matrix, class_selection)
        # selected_biases =  tf.gather_nd(bias_matrix, class_selection)
        # z2 = z * selected_weights + selected_biases
        z2 = tf.concat([z,embedding], axis = -1)
        batch_gamma = tf.layers.dense(z2, v, name='gen_gamma_projection_%i_%i'%(i,i2))
        batch_beta = tf.layers.dense(z2, v, name='gen_beta_projection_%i_%i'%(i,i2))
        batch_beta = tf.expand_dims(tf.expand_dims(batch_beta,1),1)
        _,xs,ys,_ = x_normed.get_shape()
        batch_beta = tf.tile(batch_beta,[1,xs, ys,1])
        new_x = tf.einsum('abcd,ad->abcd',x_normed, batch_gamma) + batch_beta
        new_x = tf.nn.relu(new_x)
        return new_x


def batch_norm_sota(x,classes,v, z, embedding,i,i2):
    '''manual batch norm because tensorflow can't set beta or gamma'''
    with tf.variable_scope('batch_norm_%i_%i'%(i,i2)) as scope:
        epsilon = 1e-8
        mean, variance = tf.nn.moments(x, axes=[0,1,2])
        x_normed = (x - mean) / tf.sqrt(variance + epsilon)
        z2 = tf.concat([z, embedding],-1)
        batch_gamma = tf.layers.dense(z2, v, name='gen_gamma_projection_%i_%i'%(i,i2))
        batch_beta = tf.layers.dense(z2, v, name='gen_beta_projection_%i_%i'%(i,i2))
        batch_beta = tf.expand_dims(tf.expand_dims(batch_beta,1),1)
        _,xs,ys,_ = x_normed.get_shape()
        batch_beta = tf.tile(batch_beta,[1,xs, ys,1])
        new_x = tf.einsum('abcd,ad->abcd',x_normed, batch_gamma) + batch_beta
        new_x = tf.nn.relu(new_x)
        return new_x


def batch_norm_cond(x,classes,v, embedding,i,i2):
    '''manual batch norm because tensorflow can't set beta or gamma'''
    with tf.variable_scope('batch_norm_%i_%i'%(i,i2)) as scope:
        class_max =tf.argmax(classes, axis = 1)
        # m = tf.shape(class_max)
        # m = tf.concat([m,[G_PROJECTIONS[i]]],0)
        # z = tf.random_normal(m)
        epsilon = 1e-4
        mean, variance = tf.nn.moments(x, axes=[0,1,2])
        x_normed = (x - mean) / tf.sqrt(variance + epsilon)
        class_matrix = tf.get_variable('gen_mapping_w_%i_%i'%(i,i2), [NUM_CLASSES, v])
        bias_matrix = tf.get_variable('gen_mapping_b_%i_%i'%(i,i2), [NUM_CLASSES, v])
        class_selection = tf.expand_dims(class_max, -1)
        batch_gamma = tf.gather_nd(class_matrix, class_selection)
        batch_beta =  tf.gather_nd(bias_matrix, class_selection)
        batch_beta = tf.expand_dims(tf.expand_dims(batch_beta,1),1)
        _,xs,ys,_ = x_normed.get_shape()
        batch_beta = tf.tile(batch_beta,[1,xs, ys,1])
        new_x = tf.einsum('abcd,ad->abcd',x_normed, batch_gamma) + batch_beta
        new_x = tf.nn.relu(new_x)
        return new_x

def feature_concat(x,classes):
    classes = tf.expand_dims(tf.expand_dims(classes,1),1)
    _,xs,ys,_ = x.get_shape()
    classes = tf.tile(classes,[1,xs, ys,1])
    x = tf.concat([x,classes], axis = -1)
    return x

def batch_norm_disc(x,classes,v,i,i2):
    '''manual batch norm because tensorflow can't set beta or gamma'''
    with tf.variable_scope('batch_norm_%i_%i'%(i,i2)) as scope:
        print(classes.get_shape())
        class_max =tf.argmax(classes, axis = 1)
        print(class_max.get_shape())
        epsilon = 1e-4
        mean, variance = tf.nn.moments(x, axes=[0,1,2])
        x_normed = (x - mean) / tf.sqrt(variance + epsilon)
        class_matrix = tf.get_variable('d_mapping_w_%i_%i'%(i,i2), [NUM_CLASSES, v])
        bias_matrix = tf.get_variable('d_mapping_b_%i_%i'%(i,i2), [NUM_CLASSES, v])
        class_selection = tf.expand_dims(class_max, -1)
        batch_gamma = tf.gather_nd(class_matrix, class_selection)
        batch_beta =  tf.gather_nd(bias_matrix, class_selection)
        batch_beta = tf.expand_dims(tf.expand_dims(batch_beta,1),1)
        _,xs,ys,_ = x_normed.get_shape()
        batch_beta = tf.tile(batch_beta,[1,xs, ys,1])
        new_x = tf.einsum('abcd,ad->abcd',x_normed, batch_gamma) + batch_beta
        return new_x

def spectral_norm(w, iteration=1):
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])
   u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
   u_hat = u
   v_hat = None
   for i in range(iteration):
       """
       power iteration
       Usually iteration = 1 will be enough
       """
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = tf.nn.l2_normalize(v_)
       u_ = tf.matmul(v_hat, w)
       u_hat = tf.nn.l2_normalize(u_)
   u_hat = tf.stop_gradient(u_hat)
   v_hat = tf.stop_gradient(v_hat)
   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = w / sigma
       w_norm = tf.reshape(w_norm, w_shape)
   return w_norm

def conv_spectral(convVals,v, filter_size, act=None,strides =(1,1),name=None):
    with tf.variable_scope(name) as scope:
       w = tf.get_variable("kernel", shape=[filter_size[0], filter_size[1], convVals.get_shape()[-1], v])
       b = tf.get_variable("bias", [v], initializer=tf.constant_initializer(0.0))
       x = tf.nn.conv2d(input=convVals, filter=spectral_norm(w), strides=[1, strides[0], strides[0], 1], padding="SAME") + b
    return x
