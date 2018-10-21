from __future__ import print_function
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *


IMAGE_SIZE = 28,28,1
CONVOLUTIONS = [-32, -64, 128]
D_HIDDEN_SIZE = 1000
NUM_CLASSES = 10
LEARNING_RATE = 1e-4
MOMENTUM = 0
BATCH_SIZE = 256
TEST_BATCH_SIZE = 10000
FILEPATH = '/Data/FashionMNIST/'
TRAIN_INPUT = FILEPATH + 'train-images-idx3-ubyte'
TRAIN_LABEL = FILEPATH + 'train-labels-idx1-ubyte'
TEST_INPUT = FILEPATH + 't10k-images-idx3-ubyte'
TEST_LABEL = FILEPATH + 't10k-labels-idx1-ubyte'
PERM_MODEL_FILEPATH = '/Models/LabelFashionMNIST/model.ckpt' #filepaths to model and summaries
SUMMARY_FILEPATH ='/Models/LabelFashionMNIST/summaries/'

RESTORE = True
WHEN_SAVE = 2000
WHEN_TEST = 100
ITERATIONS = 1000000

def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 1])
    return image / 255.0

def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)
    label = tf.cast(label, tf.int32)
    label = tf.reshape(label, [1])
    return  label

def return_datatset_train():
    images = tf.data.FixedLengthRecordDataset(
      TRAIN_INPUT, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 1, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TRAIN_LABEL, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))

def return_datatset_test():
    images = tf.data.FixedLengthRecordDataset(
      TEST_INPUT, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 1, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TEST_LABEL, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def create_discriminator(x_image, reuse = False):
    '''Create a discrimator, not the convolutions may be negative to represent
        downsampling'''

    with tf.variable_scope("main_labeler") as scope:
        if reuse: #get previous variable if we are reusing the discriminator but with fake images
            scope.reuse_variables()
        xs, ys = IMAGE_SIZE[0],IMAGE_SIZE[1]
        inputs = InputLayer(x_image, name='inputs')

        convVals = inputs
        res = inputs
        for i,v in enumerate(CONVOLUTIONS):
            '''Similarly tile for constant reference to class'''
            if v < 0:
                v *= -1
                convVals = Conv2d(convVals,v, (3, 3), act=tf.nn.relu,strides =(2,2),name='conv_0_%i'%(i))
            else:
                convVals = Conv2d(convVals,v, (3, 3), act=tf.nn.relu,strides =(1,1),name='conv_0_%i'%(i))
            # batch = BatchNormLayer(convVals,act=tf.nn.relu, is_train=True,name='bn_1_%i'%(i))
            conv = Conv2d(convVals,abs(v), (3, 3), strides =(1,1),name='conv_1_%i'%(i))
            # batch = BatchNormLayer(conv,act=tf.nn.relu, is_train=True,name='bn_2_%i'%(i))
            conv = Conv2d(conv,abs(v), (3, 3), strides =(1,1),name='conv_2_%i'%(i))
            convVals = InputLayer(convVals.outputs + conv.outputs, name='res_sum_%i'%(i))

        flat3 = FlattenLayer(convVals, name = 'flatten')
        hid3 = DenseLayer(flat3, D_HIDDEN_SIZE,act = tf.nn.relu, name = 'fcl')
        y_conv = DenseLayer(hid3, NUM_CLASSES,  name = 'output').outputs
        return y_conv

def build_model(x, classes,reuse=False):
    # classes = tf.squeeze(classes, axis = 1)
    classes = tf.squeeze(classes, 1)
    classes_one =tf.one_hot(classes, NUM_CLASSES, dtype=tf.float32)
    prefix = 'train_'
    if reuse:
        prefix='test_'
    y_conv = create_discriminator(x,reuse) #real image discrimator
    with tf.variable_scope('logistics') as scope:
        if reuse: #get previous variable if we are reusing the discriminator but with fake images
            scope.reuse_variables()

        d_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=classes_one, logits=y_conv))
        d_cross_entropy_summary = tf.summary.scalar(prefix + 'loss',d_cross_entropy)
        final_true = tf.argmax(y_conv,axis= 1,output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(final_true, classes), tf.float32))#determine various accuracies
        accuracy_summary = tf.summary.scalar(prefix + 'accuracy',accuracy)
        if not reuse:
            train_step = tf.train.AdamOptimizer(LEARNING_RATE,beta1=MOMENTUM).minimize(d_cross_entropy)
        else:
            train_step = None
        scalar_summary = tf.summary.merge([d_cross_entropy_summary,accuracy_summary])
    return scalar_summary, train_step

def init_model(sess, RESTORE = True):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'main_labeler' in var.name] #find trainable discriminator variable
    saver_perm = tf.train.Saver(var_list = d_vars)
    if PERM_MODEL_FILEPATH is not None and RESTORE:
        print('RESTORE')
        saver_perm.restore(sess, PERM_MODEL_FILEPATH)
    else:
        print('SAVE')
        saver_perm.save(sess, PERM_MODEL_FILEPATH)
    return saver_perm




if __name__ == "__main__":
    sess = tf.Session()#start the session
    ##############GET DATA###############
    train_ship = return_datatset_train().repeat().batch(BATCH_SIZE)
    test_ship = return_datatset_test().repeat().batch(TEST_BATCH_SIZE)
    train_iterator = train_ship.make_initializable_iterator()
    test_iterator = test_ship.make_initializable_iterator()
    train_input, train_label = train_iterator.get_next()
    test_input, test_label = test_iterator.get_next()
    sess.run([train_iterator.initializer,test_iterator.initializer])
    scalar_summary, train_step = build_model(train_input, train_label)
    test_scalar_summary, _ = build_model(train_input, train_label, reuse=True)

    ##########################################################################
    #Call function to make tf models
    ##########################################################################
    sess.run(tf.global_variables_initializer())
    saver_perm = init_model(sess)
    train_writer = tf.summary.FileWriter(SUMMARY_FILEPATH,
                                  sess.graph)
    for i in range(ITERATIONS):
        input_summary_ex, _= sess.run([scalar_summary, train_step])
        train_writer.add_summary(input_summary_ex, i)
        if not i % WHEN_TEST:
            input_summary_ex= sess.run(test_scalar_summary)
            train_writer.add_summary(input_summary_ex, i)
        if not i % WHEN_SAVE:
            saver_perm.save(sess, PERM_MODEL_FILEPATH)
        # if not i % EST_ITERATIONS:
        #     print('Epoch' + str(i / EST_ITERATIONS))
