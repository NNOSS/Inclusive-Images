from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/Data/MNIST_data", one_hot=True)

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
import numpy as np
import matplotlib.pyplot as plt
BASE_X
BASE_Y
IMAGE_SHAPE
Z_SIZE
Z2_SIZE
D_CONVOLUTIONS
G_CONVOLUTIONS

def create_discriminator(x_image, classes reuse = False):
    '''Create a discrimator, not the convolutions may be negative to represent
        downsampling'''
     with tf.variable_scope("d_discriminator") as scope:
        if reuse: #get previous variable if we are reusing the discriminator but with fake images
            scope.reuse_variables()
        xs, ys = inputSize[0],inputSize[1]
        inputs = InputLayer(x_image, name='d_inputs')

        convVals = inputs
        res = inputs
        for i,v in enumerate(D_CONVOLUTIONS):
            '''Similarly tile for constant reference to class'''
            if i < len(D_CONVOLUTIONS)-1:#if it is negative, that means we pool on this step
                if v < 0:
                    v *= -1
                    strides = (2,2)
                else:
                    strides = (1,1)
                #add necessary convolutional layers
                convVals = Conv2d(convVals, v, (3, 3),strides = strides, name='d_conv1_%s'%(i))
                # fully connecter layer
        flat3 = FlattenLayer(convVals, name = 'd_flatten')
        inputClass =InputLayer(classes, name='d_class_inputs')
        concat = ConcatLayer([flat3, inputClass], 1, name ='d_concat_layer')
        hid3 = DenseLayer(concat, , act = tf.nn.relu, name = 'd_fcl')
        y_conv = DenseLayer(hid3, 1, name = 'd_hidden_encode').outputs
        return y_conv

def create_generator(self,z, classes):
    '''Function to create the enerator and give its output
    Note that in convolutions, negative convolutions mean downsampling'''
    # Generator Net
    with tf.variable_scope("gen_generator") as scope:
        class_matrix = tf.get_variable("gen_mapping_w", [Z_SIZE, Z2_SIZE, NUM_CLASSES])
        bias_matrix = tf.get_variable("gen_mapping_b", [Z2_SIZE, NUM_CLASSES])
        class_selection = tf.argmax(classes, axis = 1)
        selected_weights = class_matrix[:,:,class_selection]
        selected_biases = bias_matrix[:, class_selection]
        z2 = z * selected_weights + selected_biases
        inputs = InputLayer(z2, name='gen_inputs')
        inputClass =InputLayer(classes, name='gen_class_inputs_z')
        numPools = len(G_CONVOLUTIONS) #count number of total convolutions
        print(numPools)
        xs, ys = IMAGE_SIZE[0]/(2**numPools), IMAGE_SIZE[1]/(2**numPools) #calculate start image size from numPools
        sizeDeconv = xs * ys * abs(G_CONVOLUTIONS[0])
        conat_layer = ConcatLayer([inputs, inputClass], 1, name ='gen_concat_layer_z2')
        deconveInputFlat = DenseLayer(conat_layer, sizeDeconv, act = tf.nn.relu, name = 'gen_fdeconv')#dense layer to input to be reshaped
        convVals = ReshapeLayer(deconveInputFlat, (-1, BASE_X, BASE_Y, abs(G_CONVOLUTIONS[0])), name = 'gen_unflatten')

        for i,v in enumerate(G_CONVOLUTIONS[1:]):#for every convolution
            '''The purpose of tile is to add outputs of correct image size
            that are all 1s to represent a certain class'''
            batch = BatchNormlayer(convVals,act=tf.nn.relu, is_trian=True,name='gen_bn_1_%i'%(i))
            conv = Conv2d(batch,v, (3, 3), stride =(1,1),name='gen_deconv_1_%i'%(i))
            batch = BatchNormlayer(conv,act=tf.nn.relu, is_trian=True,name='gen_bn_2_%i'%(i))
            conv = Conv2d(batch,v, (3, 3), stride =(1,1),name='gen_deconv_2_%i'%(i))
            convVals += conv
            convVals = UpSampling2dLayer(convVals, (2,2),name='gen_upsample_%i'%(i))
        batch = BatchNormlayer(convVals,act=tf.nn.relu, is_trian=True,name='gen_bn_1_%i'%(i))
        convVals = Conv2d(convVals,3, (3, 3), stride = (1,1), act=tf.nn.tanh,name='gen_fake_input'))
        return convVals.outputs #return flattened outputs

def create_model(x, classes, y):
    y_conv = createDiscriminator(x,classes) #real image discrimator
    convolutions = [1] + convolutions
    fake_classes =
    fake_input = createGenerator(z,classes) #generator
    fake_y_conv = createDiscriminator(fake_input,classes,reuse = True)#fake image discrimator
    fake_input_summary = tf.summary.image("fake_inputs", fake_input,max_outputs = 6)#show fake image
    real_input_summary = tf.summary.image("real_inputs", x,max_outputs = 6)#show real image
    t_vars = tf.trainable_variables()
    print(t_vars)

    d_vars = [var for var in t_vars if 'd_' in var.name] #find trainable discriminator variable
    print(d_vars)
    gen_vars = [var for var in t_vars if 'gen_' in var.name] #find trainable discriminator variable
    print(gen_vars)

    d_cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)) + tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=fake_y_, logits=fake_y_conv))# reduce mean for discriminator
    d_cross_entropy_summary = tf.summary.scalar('d_loss',d_cross_entropy)
    train_step = tf.train.AdamOptimizer(2e-4,beta1=.9).minimize(d_cross_entropy, var_list = d_vars)

    accuracy_real = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)), tf.float32))#determine various accuracies
    accuracy_fake = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fake_y_conv,1), tf.argmax(fake_y_,1)), tf.float32))
    accuracy_summary_real = tf.summary.scalar('accuracy_real',accuracy_real)
    accuracy_summary_fake = tf.summary.scalar('accuracy_fake',accuracy_fake)

    gen_cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=gen_y_, logits=fake_y_conv))#reduce mean for generator
    gen_train_step = tf.train.AdamOptimizer(2e-4,beta1=.9).minimize(gen_cross_entropy,var_list = gen_vars)
    gen_cross_entropy_summary = tf.summary.scalar('g_loss',gen_cross_entropy)
