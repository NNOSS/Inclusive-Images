import sys
sys.path.append('./../FashionGAN')
import os
import tensorflow as tf
import numpy as np
from tensorlayer.layers import *
# import testerModel
from projectionMethods import *

MODELS_NAME = ['Classical','Projection', 'Ours', 'Class BN']
WHICH_MODEL = 1
BASE_X = 64/16
BASE_Y = 64/16
IMAGE_SIZE = 64, 64,3
Z_SIZE = 120
D_CH = 64
D_CONVOLUTIONS = [-1*D_CH, -2*D_CH, -4*D_CH,-8*D_CH]
D_TRANSFORM = [0,0,0,0]

# D_HIDDEN_SIZE = 1000
D_EMBED_SIZE = 256
G_CH = 64
G_CONVOLUTIONS = [-8*G_CH, -4*G_CH,-2*G_CH, -1*G_CH]
G_TRANSFORM = [0,0,0,0]
G_EMBED_SIZE = 6cd
NUM_CLASSES = 2
D_LEARNING_RATE = 2e-4
G_LEARNING_RATE = 1e-4
MOMENTUM = 0
BATCH_SIZE = 64
# FILEPATH = '/Data/FashionMNIST/'
ssh = False
IMAGENET_PATH = '/Data/Imagenet/DogsvCats/train/'
TRAIN_INPUT_SAVE = '/Data/Imagenet/DogsvCats/train_images_64'
TRAIN_LABEL_SAVE = '/Data/Imagenet/DogsvCats/train_labels_64'
PERM_MODEL_FILEPATH = '/Models/ImageNet/DC_Trans/model.ckpt'
SUMMARY_FILEPATH = '/Models/ImageNet/DC_Trans/Summaries/'
if ssh:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    IMAGENET_PATH = '/Data/Imagenet/DogsvCats/train/'
    TRAIN_INPUT_SAVE = '/data2/user_data/gsteelman/Data/Imagenet/DogsvCats/train_images'
    TRAIN_LABEL_SAVE = '/data2/user_data/gsteelman/Data/Imagenet/DogsvCats/train_labels'
    PERM_MODEL_FILEPATH = '/data2/user_data/gsteelman/Models/ImageNet/DC_Trans/model.ckpt'
    SUMMARY_FILEPATH = '/data2/user_data/gsteelman/Models/ImageNet/DC_Trans/Summaries/'


RESTORE = False
WHEN_DISP = 50
# WHEN_TEST = 50
NUM_OUTPUTS = 10
WHEN_SAVE = 1000
ITERATIONS = 100000

def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]])
    return image / 255.0

def decode_label(label):
    label = tf.decode_raw(label, tf.int32)
    label = tf.reshape(label, [1])
    return  label

def return_datatset_train():
    images = tf.data.FixedLengthRecordDataset(
      TRAIN_INPUT_SAVE, IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2]).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TRAIN_LABEL_SAVE, 1*4).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def create_discriminator(x_image, classes, reuse = False):
    '''Create a discrimator, not the convolutions may be negative to represent
        downsampling'''
    with tf.variable_scope("d_discriminator") as scope:
        if reuse: #get previous variable if we are reusing the discriminator but with fake images
            scope.reuse_variables()
        embedding = DenseLayer(InputLayer(classes), D_EMBED_SIZE,  name = 'd_embed').outputs
        # embedding = embedding / tf.reduce_sum(tf.norm(embedding, axis=1))

        xs, ys = IMAGE_SIZE[0],IMAGE_SIZE[1]
        convVals = x_image
        for i,v in enumerate(D_CONVOLUTIONS):
            '''Similarly tile for constant reference to class'''
            if D_TRANSFORM[i]:
                convVals = transformer_sota(InputLayer(convVals), 1, v, i).outputs
                continue

                # convVals = conv_spectral(convVals,v, (5, 5), act=tf.nn.relu,strides =(2,2),name='d_conv_0_%i'%(i))
            # else:
            #     convVals = conv_spectral(convVals,v, (3, 3), act=tf.nn.relu,strides =(1,1),name='d_conv_0_%i'%(i))
            # if WHICH_MODEL ==0:
            #     batch = BatchNormLayer(convVals,act=tf.nn.relu, is_train=True,name='d_bn_1_%i'%(i))
            #     batch = feature_concat(batch.outputs, classes)
            # else:
            # batch = batch_norm_disc(convVals,classes, v, 0, i)
            batch = tf.nn.relu(convVals)
            conv = conv_spectral(batch,abs(v), (3, 3), strides =(1,1),name='d_conv_1_%i'%(i))
            # if WHICH_MODEL ==0:
            #     batch = BatchNormLayer(convVals,act=tf.nn.relu, is_train=True,name='d_bn_1_%i'%(i))
            #     batch = feature_concat(batch.outputs, classes)
            # else:
            # batch = batch_norm_disc(conv,classes, v, 1, i)
            batch = tf.nn.relu(conv)
            conv = conv_spectral(batch,abs(v), (3, 3), strides =(1,1),name='d_conv_2_%i'%(i))
            convVals = conv_spectral(convVals,abs(v), (1, 1), act=tf.nn.relu,strides =(1,1),name='d_conv_0_%i'%(i))

            if v < 0:
                v *= -1
                convVals = tf.nn.pool(convVals, (2, 2), "AVG",'SAME',strides =(2,2),name='res_downsample_%i'%(i))
                conv = tf.nn.pool(conv, (2, 2), "AVG",'SAME',strides =(2,2),name='downsample_%i'%(i))
            convVals = convVals + conv

            #add necessary convolutional layers
                # fully connecter layer
        flat = tf.reduce_sum(tf.reduce_sum(tf.nn.relu(convVals),axis=1),axis=1)
        flatI= InputLayer(flat)
        # flat3 = FlattenLayer(convVals, name = 'd_flatten')
        # inputClass =InputLayer(classes, name='d_class_inputs')
        # concat = ConcatLayer([flat3, inputClass], 1, name ='d_concat_layer')
        # hid3 = DenseLayer(flat3, D_HIDDEN_SIZE,act = tf.nn.relu, name = 'd_fcl')
        # hid3 = DropoutLayer(hid3, keep=0.85, is_fix=True,name='drop')
        y_conv = DenseLayer(flatI, 1,  name = 'd_output').outputs
        # flat = flat / tf.reduce_sum(tf.norm(flat, axis=1))
        effect = tf.reduce_sum( tf.multiply( flat[:, :D_EMBED_SIZE], embedding ), 1, keep_dims=True )
        return y_conv + effect

def create_generator(z, classes):
    '''Function to create the enerator and give its output
    Note that in convolutions, negative convolutions mean downsampling'''
    # Generator Net
    with tf.variable_scope("gen_generator") as scope:
        embedding = DenseLayer(InputLayer(classes), G_EMBED_SIZE,  name = 'gen_embed').outputs
        # embedding = embedding / tf.reduce_sum(tf.norm(embedding, axis=1))

        if WHICH_MODEL == 2:
            class_matrix = tf.get_variable("gen_mapping_w", [NUM_CLASSES, Z_SIZE])
            bias_matrix = tf.get_variable("gen_mapping_b", [NUM_CLASSES, Z_SIZE])
            class_selection = tf.expand_dims(tf.argmax(classes, axis = 1), -1)
            selected_weights = tf.gather_nd(class_matrix, class_selection)
            selected_biases =  tf.gather_nd(bias_matrix, class_selection)
            z = z * selected_weights + selected_biases
        splits = len(G_CONVOLUTIONS) + 1
        z_cut = tf.reshape(z,(-1,splits,Z_SIZE/splits))
        inputs = InputLayer(z_cut[:,0,:], name='gen_inputs')
        inputClass =InputLayer(classes, name='gen_class_inputs_z')
        numPools = sum([1 for i in G_CONVOLUTIONS if i < 0]) #count number of total convolutions
        print(numPools)
        sizeDeconv = BASE_X * BASE_X * abs(G_CONVOLUTIONS[0])
        # conat_layer = ConcatLayer([inputs, inputClass], 1, name ='gen_concat_layer_z2')
        # print(conat_layer.get_shape())
        deconveInputFlat = DenseLayer(inputs, sizeDeconv, name = 'gen_fdeconv')#dense layer to input to be reshaped
        convVals = ReshapeLayer(deconveInputFlat, (-1, BASE_X, BASE_Y, abs(G_CONVOLUTIONS[0])), name = 'gen_unflatten')

        for i,v in enumerate(G_CONVOLUTIONS):#for every convolution
            if G_TRANSFORM[i]:
                convVals = transformer_sota(convVals, 1, v, i)
                continue
            z_current = z_cut[:,i+1,:]
            if v < 0:
                v *= -1
                convVals = UpSampling2dLayer(convVals, (2,2),name='gen_upsample_%i'%(i))
            batch = return_rep(WHICH_MODEL,convVals,classes, convVals.outputs.get_shape()[3], z_current, embedding, 0,i)
            batch = InputLayer(batch)
            conv = Conv2d(batch,abs(v), (3, 3), strides =(1,1),name='gen_deconv_1_%i'%(i))
            batch = return_rep(WHICH_MODEL,conv,classes, v, z_current, embedding, 1,i)
            batch = InputLayer(batch)
            conv = Conv2d(batch,abs(v), (3, 3), strides =(1,1),name='gen_deconv_2_%i'%(i))
            convVals = Conv2d(convVals,v, (1, 1), strides =(1,1),name='gen_deconv_0_%i'%(i))
            convVals = InputLayer(convVals.outputs + conv.outputs, name='res_sum_%i'%(i))
        batch = BatchNormLayer(convVals,act=tf.nn.relu, is_train=True,name='gen_bn_final')
        # batch = UpSampling2dLayer(batch, (2,2),name='gen_upsample_final')
        convVals = Conv2d(batch,IMAGE_SIZE[2], (3, 3), strides = (1,1), act=tf.nn.tanh,name='gen_fake_input')
        return convVals.outputs #return flattened outputs

def build_model(x, og_classes):
    # classes = tf.squeeze(classes, axis = 1)
    classes = tf.squeeze(og_classes, 1)
    classes_one =tf.one_hot(classes, NUM_CLASSES)
    fake_classes_one = tf.one_hot(classes, NUM_CLASSES)
    y = tf.expand_dims(tf.ones_like(classes, dtype=tf.float32),-1)
    fake_y = tf.expand_dims(tf.zeros_like(classes, dtype=tf.float32),-1)
    y_conv = create_discriminator(x,classes_one) #real image discrimator

        # print(classes.get_shape())
        # m = classes.get_shape()
    m = tf.shape(classes)
    m = tf.concat([m,[Z_SIZE]],0)
    z = tf.truncated_normal(m)

    fake_input = create_generator(z,fake_classes_one) #generator
    # test_summary, _ = testerModel.build_model(fake_input, tf.cast(og_classes,dtype=tf.int32))
    print(fake_input.get_shape())
    fake_y_conv = create_discriminator(fake_input,fake_classes_one,reuse = True)#fake image discrimator
    with tf.variable_scope('logistics') as scope:
        fake_input_summary = tf.summary.image("fake_inputs", tf.nn.relu(fake_input),max_outputs = NUM_OUTPUTS)#show fake image
        real_input_summary = tf.summary.image("real_inputs", x,max_outputs = NUM_OUTPUTS)#show real image
        t_vars = tf.trainable_variables()
    # print(t_vars)

        d_vars = [var for var in t_vars if 'd_' in var.name] #find trainable discriminator variable
        print(d_vars)
        gen_vars = [var for var in t_vars if 'gen_' in var.name] #find trainable discriminator variable
        # print(gen_vars)
        print(y.get_shape())
        print(y_conv.get_shape())

        d_cross_entropy = tf.reduce_mean(tf.nn.relu(1-y_conv)) + tf.reduce_mean(tf.nn.relu(1+fake_y_conv))
        # tf.reduce_mean(
        #         tf.losses.hinge_loss(labels=fake_y, logits=fake_y_conv))# reduce mean for discriminator
        d_cross_entropy_summary = tf.summary.scalar('d_loss',d_cross_entropy)
        gen_cross_entropy =  -1 *tf.reduce_mean(fake_y_conv)
        y_conv = tf.sigmoid(y_conv)
        fake_y_conv = tf.sigmoid(fake_y_conv)
        final_true = tf.round(y_conv)
        final_false = tf.round(fake_y_conv)
        accuracy_real = tf.reduce_mean(tf.cast(tf.equal(final_true, y), tf.float32))#determine various accuracies
        accuracy_fake = tf.reduce_mean(tf.cast(tf.equal(final_false,fake_y), tf.float32))
        accuracy_summary_real = tf.summary.scalar('accuracy_real',accuracy_real)
        accuracy_summary_fake = tf.summary.scalar('accuracy_fake',accuracy_fake)
    train_step = tf.train.AdamOptimizer(D_LEARNING_RATE,beta1=MOMENTUM).minimize(d_cross_entropy, var_list = d_vars)
    gen_train_step = tf.train.AdamOptimizer(G_LEARNING_RATE,beta1=MOMENTUM).minimize(gen_cross_entropy,var_list = gen_vars)
    gen_cross_entropy_summary = tf.summary.scalar('g_loss',gen_cross_entropy)
    scalar_summary = tf.summary.merge([d_cross_entropy_summary,gen_cross_entropy_summary,accuracy_summary_real,accuracy_summary_fake])
    image_summary = tf.summary.merge([real_input_summary,fake_input_summary])
    return scalar_summary, image_summary, train_step, gen_train_step

def train_model():
    sess = tf.Session()#start the session
    ##############GET DATA###############
    train_ship = return_datatset_train().repeat().batch(BATCH_SIZE)
    train_iterator = train_ship.make_initializable_iterator()
    train_input, train_label = train_iterator.get_next()
    sess.run([train_iterator.initializer])
    scalar_summary, image_summary, train_step, gen_train_step = build_model(train_input, train_label)

    ##########################################################################
    #Call function to make tf models
    ##########################################################################
    sess.run(tf.global_variables_initializer())
    t_vars = tf.trainable_variables()
    # _ =testerModel.init_model(sess)

    # d_vars = [var for var in t_vars if 'd_' in var.name] #find trainable discriminator variable
    # gen_vars = [var for var in t_vars if 'gen_' in var.name]
    saver_perm = tf.train.Saver()
    if PERM_MODEL_FILEPATH is not None and RESTORE:
        saver_perm.restore(sess, PERM_MODEL_FILEPATH)
    else:
        print('SAVE')
        saver_perm.save(sess, PERM_MODEL_FILEPATH)
    train_writer = tf.summary.FileWriter(SUMMARY_FILEPATH,
                                  sess.graph)
    trains = [train_step,train_step,gen_train_step]
    for i in range(ITERATIONS):
        train = trains[i%3]
        if not i % WHEN_DISP:
            input_summary_ex, image_summary_ex,_= sess.run([scalar_summary, image_summary, train])
            train_writer.add_summary(image_summary_ex, i)
        else:
            input_summary_ex, _= sess.run([scalar_summary, train])
        train_writer.add_summary(input_summary_ex, i)

        # if not i % WHEN_TEST:
        #     test_summary_ex= sess.run(test_summary)
        #     train_writer.add_summary(test_summary_ex, i)

        if not i % WHEN_SAVE:
            saver_perm.save(sess, PERM_MODEL_FILEPATH)
        # if not i % EST_ITERATIONS:
        #     print('Epoch' + str(i / EST_ITERATIONS))

if __name__ == "__main__":
    NUM_TRIALS = 5
    train_model()
    #
    # for i in range(NUM_TRIALS):
    #     for j in range(len(MODELS_NAME)):
    #         WHICH_MODEL = j
    #         PERM_MODEL_FILEPATH = '/Models/FashionMNIST/'+ MODELS_NAME[WHICH_MODEL]\
    #             + '/Trial'+ str(i)+'/model.ckpt' #filepaths to model and summaries
    #         SUMMARY_FILEPATH = '/Models/FashionMNIST/'+ MODELS_NAME[WHICH_MODEL]\
    #             + '/Trial'+ str(i)+'/Summaries/' #filepaths to model and summaries
    #         train_model()
    #         tf.reset_default_graph()
