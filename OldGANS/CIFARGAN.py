import sys
sys.path.append('./../FashionGAN')
import os
import tensorflow as tf
import numpy as np
from tensorlayer.layers import *
# import testerModel
from projectionMethods import *

MODELS_NAME = ['Classical','Projection', 'Ours', 'Class BN']
WHICH_MODEL = 2
BASE_X = 32/8
BASE_Y = 32/8
IMAGE_SIZE = 32, 32,3
Z_SIZE = 60
D_CONVOLUTIONS = [-64, -128, -256,512]
D_TRANSFORM = [0,0,0,0]

# D_HIDDEN_SIZE = 1000
D_EMBED_SIZE = 128
G_CONVOLUTIONS = [512, -256,-128, -64]
G_TRANSFORM = [0,0,0,0]
G_EMBED_SIZE = 48
NUM_CLASSES = 10
D_LEARNING_RATE = 0
G_LEARNING_RATE = 0
MOMENTUM = 0
BATCH_SIZE = 128

FILEPATH = '/Data/OldData/CIFAR10/cifar-10-batches-bin/'
TRAIN_INPUT = [FILEPATH + 'data_batch_1.bin', FILEPATH + 'data_batch_2.bin',
    FILEPATH + 'data_batch_3.bin', FILEPATH + 'data_batch_4.bin', FILEPATH + 'data_batch_5.bin']
PERM_MODEL_FILEPATH = '/Models/CIFAR10/model.ckpt' #filepaths to model and summaries
SUMMARY_FILEPATH ='/Models/CIFAR10/Summaries/'

RESTORE = True
WHEN_DISP = 50
# WHEN_TEST = 50
NUM_OUTPUTS = 30
WHEN_SAVE = 100
ITERATIONS = 100000


def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    label = image[0]
    image = tf.cast(image[1:], tf.float32)
    image = tf.reshape(image, [3, IMAGE_SIZE[0], IMAGE_SIZE[1]])
    image = tf.transpose(image, [1,2,0])
    return image / 255.0, label


def return_datatset_train():
    images = tf.data.FixedLengthRecordDataset(
      TRAIN_INPUT, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3 + 1).map(decode_image)
    return images#tf.data.Dataset.zip((images, labels))

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

        # if WHICH_MODEL == 2:
        #     class_matrix = tf.get_variable("gen_mapping_w", [NUM_CLASSES, Z_SIZE])
        #     bias_matrix = tf.get_variable("gen_mapping_b", [NUM_CLASSES, Z_SIZE])
        #     class_selection = tf.expand_dims(tf.argmax(classes, axis = 1), -1)
        #     selected_weights = tf.gather_nd(class_matrix, class_selection)
        #     selected_biases =  tf.gather_nd(bias_matrix, class_selection)
        #     z = z * selected_weights + selected_biases
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
        convVals = ReshapeLayer(deconveInputFlat, (-1, BASE_X, BASE_Y, abs(G_CONVOLUTIONS[0])), name = 'gen_unflatten').outputs

        for i,v in enumerate(G_CONVOLUTIONS):#for every convolution
            if G_TRANSFORM[i]:
                convVals = transformer_sota(convVals, 1, v, i)
                continue
            z_current = z_cut[:,i+1,:]
            if v < 0:
                v *= -1
                convVals = UpSampling2dLayer(InputLayer(convVals), (2,2),name='gen_upsample_%i'%(i)).outputs
            batch = return_rep(WHICH_MODEL,convVals,classes, convVals.get_shape()[3], z_current, embedding, 0,i)
            conv = conv_spectral(batch,abs(v), (3, 3), strides =(1,1),name='gen_deconv_1_%i'%(i))
            batch = return_rep(WHICH_MODEL,conv,classes, v, z_current, embedding, 1,i)
            conv = conv_spectral(batch,abs(v), (3, 3), strides =(1,1),name='gen_deconv_2_%i'%(i))
            convVals = conv_spectral(convVals,abs(v), (1, 1), strides =(1,1),name='gen_deconv_0_%i'%(i))
            convVals = convVals + conv
        batch = BatchNormLayer(InputLayer(convVals),act=tf.nn.relu, is_train=True,name='gen_bn_final')
        # batch = UpSampling2dLayer(batch, (2,2),name='gen_upsample_final')
        convVals = Conv2d(batch,IMAGE_SIZE[2], (3, 3), strides = (1,1), act=tf.nn.tanh,name='gen_fake_input')
        return convVals.outputs #return flattened outputs

def build_model(x, og_classes):
    # classes = tf.squeeze(classes, axis = 1)
    og_classes = tf.expand_dims(og_classes,-1)
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

def test_model():
    import compareImages
    FILEPATH = '/Data/OldData/CIFAR10/cifar-10-batches-bin/'
    TRAIN_INPUT = [FILEPATH + 'data_batch_1.bin', FILEPATH + 'data_batch_2.bin',
        FILEPATH + 'data_batch_3.bin', FILEPATH + 'data_batch_4.bin', FILEPATH + 'data_batch_5.bin']
    arr, ogarr = compareImages.read_data(TRAIN_INPUT)
    sess = tf.Session()#start the session
    batch = 512
    number_each_class = 10
    # z = tf.placeholder(tf.float32,(number_each_class,Z_SIZE))
    classes = tf.placeholder(tf.int32, (batch))
    m = tf.shape(classes)
    m = tf.concat([m,[Z_SIZE]],0)
    z = tf.truncated_normal(m)
    classes_one_hot= tf.one_hot(classes, NUM_CLASSES)
    fake_input = create_generator(z,classes_one_hot)
    _ = create_discriminator(fake_input,classes_one_hot)
    output= tf.nn.relu(fake_input) * 255
    sess.run(tf.global_variables_initializer())
    saver_perm = tf.train.Saver()
    saver_perm.restore(sess, PERM_MODEL_FILEPATH)
    container = np.zeros((10 * 32, 10*32,3))
    for i in range(NUM_CLASSES):
        classes_ex = np.full(batch,i)
        classes_ex[number_each_class:] = np.random.randint(0,NUM_CLASSES,size=(batch-number_each_class))
        # z_ex = np.random.normal(size=(number_each_class,Z_SIZE))
        # zeros = np.zeros((number_each_class,Z_SIZE))
        # z_ex = np.where(np.absolute(z_ex)>2,zeros, z_ex)
        feed_dict = {classes:classes_ex}
        row = sess.run(output,feed_dict=feed_dict)
        row = row[:number_each_class]
        print(row.shape)
        container[:,i*32:(i+1)*32,:] = np.reshape(row, (32*10, 32,3))
        compareImages.compare_image(arr,row[0].astype(np.uint8),ogarr)


    from PIL import Image
    img = Image.fromarray(container.astype(np.uint8), 'RGB')
    img.show()



if __name__ == "__main__":
    NUM_TRIALS = 5
    test_model()
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
