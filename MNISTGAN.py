from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
import numpy as np
import matplotlib.pyplot as plt

restore = True #whether or not to restor the file from a source
model_filepath = './../GANModelDeep/model.ckpt' #filepaths to model and summaries
summary_filepath = './../GANModelDeep/Summaries/'

# model_filepath = './../thisworks/model.ckpt'
# summary_filepath = './../thisworks/Summaries/'

class discriminator:
    def __init__(self,inputSize,convolutions,fullyconnected,output,restore = False,fileName =None):
        '''Make the instantiator, first make a lot of constants available'''
        self.inputSize = inputSize
        self.convolutions= [1] + convolutions
        self.fullyconnected = fullyconnected
        self.output = output
        self.Zsize =30
        self.numClasses =10
        self.fileName = fileName
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))#start the session
        '''Create the placeholders, then the discriminator and Generator'''
        self.Z = tf.placeholder(tf.float32, shape=[None, self.Zsize], name='Z') #random input
        self.classes = tf.placeholder(tf.float32, shape=[None, self.numClasses], name='class_inputs') #correct class
        self.y_ = tf.placeholder(tf.float32, shape=[None, output],name = 'real_y') #placeholder for real representation
        self.fake_y_ =tf.placeholder(tf.float32, shape=[None, output], name = 'fake_y') #placeholder for fake representation
        self.gen_y_ = tf.placeholder(tf.float32, shape=[None, output], name = 'gen_y') #placeholder for generator training
        self.x = tf.placeholder(tf.float32, shape=[None, inputSize[0]*inputSize[1]*inputSize[2]], name = 'true_input') #input image
        self.y_conv = self.createDiscriminator(self.x,self.classes,inputSize,self.convolutions,fullyconnected,output) #real image discrimator
        self.convolutions = [1] + convolutions
        self.fake_input = self.createGenerator(Z = self.Z, classes = self.classes,inputSize = self.Zsize, convolutions= [-32, 64],fullyconnected = None, output = [28, 28, 1]) #generator
        self.fake_y_conv = self.createDiscriminator(self.fake_input,self.classes,inputSize,self.convolutions,fullyconnected,output,reuse = True)#fake image discrimator
        self.fake_input_summary = tf.summary.image("fake_inputs", tf.reshape(self.fake_input, [-1,inputSize[0],inputSize[1],inputSize[2]]),max_outputs = 6)#show fake image
        self.real_input_summary = tf.summary.image("real_inputs", tf.reshape(self.x, [-1,inputSize[0],inputSize[1],inputSize[2]]),max_outputs = 6)#show real image

        t_vars = tf.trainable_variables()
        print(t_vars)

        self.d_vars = [var for var in t_vars if 'd_' in var.name] #find trainable discriminator variable
        print(self.d_vars)
        self.gen_vars = [var for var in t_vars if 'gen_' in var.name] #find trainable discriminator variable
        print(self.gen_vars)

        self.d_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)) + tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.fake_y_, logits=self.fake_y_conv))# reduce mean for discriminator
        self.d_cross_entropy_summary = tf.summary.scalar('d_loss',self.d_cross_entropy)
        self.train_step = tf.train.AdamOptimizer(2e-4,beta1=.9).minimize(self.d_cross_entropy, var_list = self.d_vars)

        self.accuracy_real = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1)), tf.float32))#determine various accuracies
        self.accuracy_fake = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.fake_y_conv,1), tf.argmax(self.fake_y_,1)), tf.float32))
        self.accuracy_summary_real = tf.summary.scalar('accuracy_real',self.accuracy_real)
        self.accuracy_summary_fake = tf.summary.scalar('accuracy_fake',self.accuracy_fake)

        self.gen_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.gen_y_, logits=self.fake_y_conv))#reduce mean for generator
        self.gen_train_step = tf.train.AdamOptimizer(2e-4,beta1=.9).minimize(self.gen_cross_entropy,var_list = self.gen_vars)
        self.gen_cross_entropy_summary = tf.summary.scalar('g_loss',self.gen_cross_entropy)

        self.saver = tf.train.Saver()#saving and tensorboard
        self.sess.run(tf.global_variables_initializer())
        self.train_writer = tf.summary.FileWriter(summary_filepath,
                                      self.sess.graph)

        if fileName is not None and restore:
            self.saver.restore(self.sess, self.fileName)
        else:

            self.saver.save(self.sess, self.fileName)


    def createGenerator(self,Z, classes, inputSize,convolutions,fullyconnected,output):
        '''Function to create the enerator and give its output
        Note that in convolutions, negative convolutions mean downsampling'''
        # Generator Net
        with tf.variable_scope("gen_generator") as scope:
            inputs = InputLayer(Z, name='gen_inputs')
            inputClass =InputLayer(classes, name='gen_class_inputs_z')
            concatz = ConcatLayer([inputs, inputClass], 1, name ='gen_concat_layer_z')
            numPools = sum([1 for i in convolutions if i < 0]) #count number of total convolutions
            print(numPools)
            xs, ys = output[0]/(2**numPools), output[1]/(2**numPools) #calculate start image size from numPools
            sizeDeconv = xs * ys * abs(convolutions[0])

            if fullyconnected is not None:
                concatz = DenseLayer(concatz, fullyconnected, act = tf.nn.relu, name = 'gen_fc')

            deconveInputFlat = DenseLayer(concatz, sizeDeconv, act = tf.nn.relu, name = 'gen_fdeconv')#dense layer to input to be reshaped
            deconveInput = ReshapeLayer(deconveInputFlat, (-1, xs, ys, abs(convolutions[0])), name = 'gen_unflatten')

            convolutions.append(1)
            convVals = [deconveInput]

            for i,v in enumerate(convolutions):#for every convolution
                if i < len(convolutions)-1:
                    '''The purpose of tile is to add outputs of correct image size
                    that are all 1s to represent a certain class'''
                    class_image = tf.tile(tf.expand_dims(tf.expand_dims(classes,1),1), (1,xs,ys,1))
                    class_image = InputLayer(class_image, name='gen_class_inputs_%i'%(i))
                    if convolutions[i] < 0:#decide whether to upsample
                        convolutions[i] *= -1
                        xs *= 2
                        ys *= 2
                        stride  = (2,2)
                    else:
                        stride = (1,1)

                    if i < len(convolutions)-2:#decide when to add the tiles
                        convVals[-1] = ConcatLayer([convVals[-1], class_image], 3, name ='gen_deconv_plus_classes_%i'%(i))
                    if i == len(convolutions)-2:#if on the last step, use tanh instead of relu
                        convVals.append(DeConv2d(convVals[-1],abs(convolutions[i+1]), (5, 5), (xs,ys), stride, act=tf.nn.tanh,name='gen_fake_input'))#finally add the convlayer
                    else:
                        convVals.append(DeConv2d(convVals[-1],abs(convolutions[i+1]), (5, 5), (xs,ys), stride, act=tf.nn.relu,name='gen_deconv%i'%(i)))
            return FlattenLayer(convVals[-1]).outputs #return flattened outputs


    def createDiscriminator(self,x,classes,inputSize,convolutions,fullyconnected,output,reuse=False):

        '''Create a discrimator, not the convolutions may be negative to represent
        downsampling'''
         with tf.variable_scope("d_discriminator") as scope:
            if reuse: #get previous variable if we are reusing the discriminator but with fake images
                scope.reuse_variables()
            flatSize = inputSize[0]*inputSize[1]*inputSize[2]
            x_image = tf.reshape(x, [-1,inputSize[0],inputSize[1],inputSize[2]])#reshape image into right size
            xs, ys = inputSize[0],inputSize[1]
            numPools = sum([1 for i in convolutions if i < 0])
            print(numPools)
            inputs = InputLayer(x_image, name='d_disc_inputs')
            convVals = [inputs]#list of filters
            for i,v in enumerate(convolutions):
                '''Similarly tile for constant reference to class'''
                class_image = tf.tile(tf.expand_dims(tf.expand_dims(classes,1),1), (1,xs,ys,1))
                class_image = InputLayer(class_image, name='d_class_inputs_%i'%(i))
                if i < len(convolutions)-1:#if it is negative, that means we pool on this step
                    pool=False
                    if convolutions[i+1] < 0:
                        convolutions[i+1] *= -1
                        pool = True
                        xs, ys = xs/2, ys/2
                    #add necessary convolutional layers
                    convVals[-1] = ConcatLayer([convVals[-1], class_image], 3, name ='d_conv_plus_classes_%i'%(i))
                    if pool:
                        convVals.append(Conv2d(convVals[-1], convolutions[i+1], (5, 5),strides = (2,2), act=tf.nn.relu, name='d_conv1_%s'%(i)))
                    else:
                        convVals.append(Conv2d(convVals[-1], convolutions[i+1], (5, 5),strides = (1,1), act=tf.nn.relu, name='d_conv1_%s'%(i)))
                else:
                    # fully connecter layer
                    l,w,d = inputSize[0]/(2**numPools),inputSize[1]/(2**numPools),convolutions[-1]
                    flat3 = FlattenLayer(convVals[-1], name = 'd_flatten')
                    inputClass =InputLayer(classes, name='d_class_inputs')
                    concat = ConcatLayer([flat3, inputClass], 1, name ='d_concat_layer')
                    hid3 = DenseLayer(concat, fullyconnected, act = tf.nn.relu, name = 'd_fcl')
                    concat2 = ConcatLayer([hid3, inputClass], 1, name ='d_concat_layer_2')
                    y_conv = DenseLayer(concat2, output, act = tf.nn.relu, name = 'd_hidden_encode').outputs
            return y_conv

    def train(self,iterations,batchLen = 20):
        '''Train the model'''
        fake_batch = np.concatenate((np.ones((batchLen,1)),np.zeros((batchLen,1))), axis = 1) #vector presesenting predicitng fake value
        newBatch = np.concatenate((np.zeros((batchLen,1)),np.ones((batchLen,1))), axis = 1) # vector predicting correct value
        for i in range(iterations):
            batch = self.getbatch(batchLen)# get batch
            batch = batch[0],batch[1],newBatch
            z = np.random.normal(loc= 0, scale = 1, size = [batchLen,self.Zsize])# define random z
            if i%200 == 0:
                '''Evaluate accuracy'''
                train_accuracy_real, train_accuracy_fake = self.sess.run([self.accuracy_summary_real,self.accuracy_summary_fake],feed_dict={self.x: batch[0], self.y_: batch[2],self.Z: z, self.fake_y_: fake_batch, self.classes: batch[1]})
                self.train_writer.add_summary(train_accuracy_real,i)
                self.train_writer.add_summary(train_accuracy_fake,i)
                if i % 1000 ==0:
                    '''Save session'''
                    if self.fileName is not None:
                        self.saver.save(self.sess, self.fileName)
            train, real_input_summary, d_cross_entropy_summary= self.sess.run([self.train_step,self.real_input_summary,self.d_cross_entropy_summary],
            feed_dict={self.x: batch[0], self.y_: batch[2],self.Z: z, self.fake_y_: fake_batch, self.classes: batch[1]}) #train discrimator
            self.train_writer.add_summary(d_cross_entropy_summary, i)
            gen_train, fake_input_summary,gen_cross_entropy_summary= self.sess.run([self.gen_train_step,self.fake_input_summary,self.gen_cross_entropy_summary],
            feed_dict={self.Z: z, self.gen_y_: batch[2], self.classes: batch[1]})#train generator
            if i % 50 ==0:#push images every 50 iterations
                self.train_writer.add_summary(fake_input_summary, i+1)
                self.train_writer.add_summary(real_input_summary, i+3)
            self.train_writer.add_summary(gen_cross_entropy_summary, i+2)

    def getbatch(self,batchLen):
        batch = mnist.train.next_batch(batchLen)

        return batch


myDiscriminator = discriminator(inputSize = [28,28,1],convolutions = [-32,-64], fullyconnected = 128, output = 2, fileName = model_filepath,restore = restore)
myDiscriminator.train(100000)
