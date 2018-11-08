import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

PERM_MODEL_FILEPATH = '/Models/FashionMNIST/model.ckpt' #filepaths to model and summaries
FINAL_FILEPATH = '/Models/FashionMNIST/model.ckpt'

coarse_old_vars = tf.contrib.framework.list_variables(PERM_MODEL_FILEPATH)
with tf.Graph().as_default(), tf.Session().as_default() as sess:
    new_vars = []
    for name, shape in coarse_old_vars:
        v = tf.contrib.framework.load_variable(PERM_MODEL_FILEPATH, name)
        if 'gen_' in name:
            name = name.replace('d_','gen_')
        new_vars.append(tf.Variable(v, name=name))
    print(new_vars)
    saver = tf.train.Saver(new_vars)
    sess.run(tf.global_variables_initializer())
    saver.save(sess, FINAL_FILEPATH)
