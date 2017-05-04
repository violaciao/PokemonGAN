# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import util
from skimage.transform import resize



FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer("n_epochs", "10001",
                       "number of training epochs")
tf.flags.DEFINE_integer("n_kernels", "5",
                       "number of kernels of minibatch discrimination")
tf.flags.DEFINE_integer("kernel_dim", "5",
                       "kernel dimension in minibatch discrimination")
tf.flags.DEFINE_float("dropout_rate", 0.8, "dropout keep_prob rate")

tf.flags.DEFINE_string("outfig_name_prefix", "final_version_",
                       "Output plotting file's prefix.")
tf.flags.DEFINE_string("output_file_path", "generated_images/",
                       "Output images will be stored under this file path.")
tf.flags.DEFINE_string("saved_model_path", "saved_model_best/PokeGAN_model.ckpt",
                       "The trained model will be stored here.")

# Define minibatch discriminator
def minibatch_discriminate(inpt, w, b, num_kernels=5, kernel_dim=3):
    
    x = project(inpt, w, b)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    
    return minibatch_features


# Define residual layer
def add_residual_pre(prev_layer, z_concat=None, text_filters = None, k_h = 5, k_w = 5, hidden_text_filters = None,
                     hidden_filters = None, name_func=None):
        
        filters = prev_layer.get_shape()[3].value
        if hidden_filters == None:
            hidden_filters = filters * 4
        if text_filters == None:
            text_filters = int(filters/2)
        if hidden_text_filters == None:
            hidden_text_filters = int(filters/8)
        s = prev_layer.get_shape()[1].value
        
        bn0 = util.batch_norm(name=g_name())
        bn1 = util.batch_norm(name=g_name())
        
        low_dim = util.conv2d(util.lrelu(bn0(prev_layer)), hidden_filters, k_h=k_h, k_w=k_w, name = name_func())
        
        residual = util.deconv2d(util.lrelu(bn1(low_dim), name=name_func()), 
            [batch_size, s, s, filters], k_h=k_h, k_w=k_w, name=name_func())
        
        next_layer = prev_layer + residual
        return next_layer


def g_name():
    global g_idx
    g_idx += 1
    return 'gen_' + str(g_idx)


def d_name():
    global d_idx
    d_idx += 1
    return 'dis_' + str(d_idx)


def plotPoke(x, row, col, cap_ls, fig_name):
    f, a = plt.subplots(row, col, figsize=(col*2.5, row*1.8))
    for j in range(row):
        for i in range(col):
            a[j][i].imshow(x[i+j*col])
            a[j,i].axis('off')
            a[j,i].set_title(cap_ls[i+j*col])
    f.savefig(fig_name)
    plt.close()



def get_types(Y_matrix):
    type_ls = []
    for i in range(len(Y_matrix)):
        type_ls.append(' & '.join(np.array(types)[Y_matrix[i] != 0].tolist()))
    return type_ls


# Define ConvNet layer function
def conv2d(x, W, b, strides=2):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    return tf.nn.bias_add(x, b)


# Define Deconvnet layer function
def deconv2d(x, W, b, out_shape, strides=2):
    x = tf.nn.conv2d_transpose(x, W, out_shape, strides=[1, strides, strides, 1], 
                               padding='SAME')
    return tf.nn.bias_add(x, b)


# Define fully connected layer function
def project(x, W, b):
    return tf.add(tf.matmul(x, W), b)


# Define leaky rectified linear unit (ReLu)
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


# Define dropout unit
def dropout(x, rate=0.5):
    return tf.nn.dropout(x, keep_prob=rate)


# Define generate random vector for generator
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n]) 


###############################################################
# Build genderator and discriminator structure of Pokemon GAN #
###############################################################

def generator(z):
    bs = z.get_shape()[0].value
    g_bn0 = util.batch_norm(name=g_name())
    g_bn1 = util.batch_norm(name=g_name())
    hidden_g1 = project(z, weights['gen_h1'], biases['gen_h1'])
    hidden_g1 = tf.reshape(hidden_g1, [-1, 5, 5, n_channel3])
    
    output_dim2 = tf.stack([bs, 10, 10, n_channel2])
    hidden_g2 = tf.nn.relu(dropout(g_bn0(deconv2d(hidden_g1, weights['gen_h2'], biases['gen_h2'], output_dim2)), FLAGS.dropout_rate) )
    hidden_g2.set_shape([bs, 10, 10, n_channel2])
    hidden_g2 = add_residual_pre(hidden_g2, name_func = g_name)
    
    output_dim3 = tf.stack([bs, 20, 20, n_channel1])
    hidden_g3 = tf.nn.relu(dropout(g_bn1(deconv2d(hidden_g2, weights['gen_h3'], biases['gen_h3'], output_dim3)), FLAGS.dropout_rate) )
    hidden_g3.set_shape([bs, 20, 20, n_channel1])

    hidden_g3 = add_residual_pre(hidden_g3, name_func = g_name)
    
    hidden_dae = tf.reshape(hidden_g3, [bs, -1])
    X_rec = project(hidden_dae, weights['gen_dae'], biases['gen_dae'])
    DAE_loss = tf.reduce_mean(tf.square(X_rec - z))
    
    output_dim4 = tf.stack([tf.shape(z)[0], 40, 40, 3])
    hidden_g4 = tf.nn.tanh(deconv2d(hidden_g3, weights['gen_h4'], biases['gen_h4'], output_dim4))
    return hidden_g4, DAE_loss


def discriminator(x, y):
    hidden_d1 = lrelu(dropout(conv2d(x, weights['dis_h1'], biases['dis_h1']), FLAGS.dropout_rate) )
    hidden_d2 = lrelu(dropout(conv2d(hidden_d1, weights['dis_h2'], biases['dis_h2']), FLAGS.dropout_rate) )
    hidden_d3 = lrelu(dropout(conv2d(hidden_d2, weights['dis_h3'], biases['dis_h3']), FLAGS.dropout_rate) )
    hidden_d3 = tf.reshape(hidden_d3, [-1, 5*5*n_disc_3])
    
    mini = minibatch_discriminate(tf.reshape(x, [batch_size, -1]), w = weights['dis_mini'], b = biases['dis_mini'],
                                  num_kernels=n_kernels, kernel_dim=kernel_dim) * 1

    hidden_d3_y = tf.concat([hidden_d3, y, mini], 1)
    
    hidden_d4 = lrelu(dropout(project(hidden_d3_y, weights["dis_h4"], biases['dis_h4']), FLAGS.dropout_rate) )
    hidden_d5 = project(hidden_d4, weights["dis_h5"], biases["dis_h5"])
    
    return hidden_d5

##########################################################




# Parameters
learning_rate = 1e-5
beta1 = .5
training_epochs = FLAGS.n_epochs 
batch_size = 64
display_step = 5
examples_to_show = 9
kernel_dim = FLAGS.kernel_dim
n_kernels = FLAGS.n_kernels


# Network Parameters
n_input = [40, 40, 3] # Pokemon data input (img shape: 40*40*3)
n_channel1 = 32
n_channel2 = 128  #64
n_channel3 = 256  #128
n_disc_2 = 50
n_disc_3 = 100
n_channel4 = 35
gen_dim = 5 
type_dim = 18

# tf Graph input
X = tf.placeholder(tf.float32, [batch_size]+n_input) 
Z = tf.placeholder(tf.float32, [batch_size, gen_dim])
Y = tf.placeholder(tf.float32, [batch_size, type_dim])
Z_plus_Y = tf.concat([Z, Y], axis=1)


# Store layers weights & biases
weights = {
    'dis_h1': tf.Variable(tf.truncated_normal([5, 5, 3, n_channel1], stddev=0.01)),
    'dis_h2': tf.Variable(tf.truncated_normal([5, 5, n_channel1, n_disc_2], stddev=0.01)),
    'dis_h3': tf.Variable(tf.truncated_normal([5, 5, n_disc_2, n_disc_3], stddev=0.01)),
    'dis_h4': tf.Variable(tf.truncated_normal([5*5*n_disc_3+type_dim+kernel_dim, n_channel4], stddev=0.01)),
    'dis_h5': tf.Variable(tf.truncated_normal([n_channel4, 1], stddev=0.01)),

    'dis_mini': tf.Variable(tf.truncated_normal([40*40*3, n_kernels * kernel_dim], stddev=0.01)),

    'gen_h1': tf.Variable(tf.truncated_normal([gen_dim + type_dim, 5*5*n_channel3], stddev=0.01)),
    'gen_h2': tf.Variable(tf.truncated_normal([5, 5, n_channel2, n_channel3], stddev=0.01)),
    'gen_h3': tf.Variable(tf.truncated_normal([5, 5, n_channel1, n_channel2], stddev=0.01)),
    'gen_dae': tf.Variable(tf.truncated_normal([20*20*n_channel1, gen_dim+type_dim], stddev=0.01)),
    'gen_h4': tf.Variable(tf.truncated_normal([5, 5, 3, n_channel1], stddev=0.01))
}

biases = {
    'dis_h1': tf.Variable(tf.truncated_normal([n_channel1], stddev=0.01)),
    'dis_h2': tf.Variable(tf.truncated_normal([n_disc_2], stddev=0.01)),
    'dis_h3': tf.Variable(tf.truncated_normal([n_disc_3], stddev=0.01)),
    'dis_h4': tf.Variable(tf.truncated_normal([n_channel4], stddev=0.01)),
    'dis_h5': tf.Variable(tf.truncated_normal([1], stddev=0.01)),
    
    'dis_mini': tf.Variable(tf.truncated_normal([n_kernels * kernel_dim], stddev=0.01)),
    
    'gen_h1': tf.Variable(tf.truncated_normal([5*5*n_channel3], stddev=0.01)),
    'gen_h2': tf.Variable(tf.truncated_normal([n_channel2], stddev=0.01)),
    'gen_h3': tf.Variable(tf.truncated_normal([n_channel1], stddev=0.01)),
    'gen_dae': tf.Variable(tf.truncated_normal([gen_dim + type_dim], stddev=0.01)),
    
    'gen_h4': tf.Variable(tf.truncated_normal([3], stddev=0.01))
}


path = os.getcwd()

g_idx = 0
d_idx = 0

df = pd.read_csv('Pokemon.csv', index_col=None, names=None, usecols=['Name', 'Type 1', 'Type 2'])
df.sort_values(by='Name', inplace=True)
df.set_index('Name', inplace=True)
df.columns=['type1', 'type2']
type1_dummies = pd.get_dummies(df.type1, prefix=None)
type2_dummies = pd.get_dummies(df.type2, prefix=None)
type_dummies = type1_dummies + type2_dummies

# DataFrame of types
df_type = pd.concat([type_dummies], axis=1)
types = list(df_type.columns)

# Matrix of types
poke_type = df_type.as_matrix()

# Get the duo-type list
typeDuo_list = []
for i in range(len(df_type)):
    row = df_type.iloc[i,:]
    if row.sum() == 2:
        typeDuo_list.append(row.values.tolist())
        
typeDuo = np.vstack({tuple(row) for row in typeDuo_list})[:46]

# Define the output plotting types and captions
UserType = np.concatenate((np.eye(18), typeDuo), 0)
caps = get_types(UserType)


# Create an empty array to store pokemon pics
orig_img = np.empty((0, 40, 40, 3), dtype='float32')
orig_type = np.empty((0, 18), dtype='float32')
im_list = []
kk = 0


# Load all images and append into orig_img
# Load all images'types and append into orig_type

for pic in glob.glob(path+'/data/Pokemon/*.png'):
    img = mpimg.imread(pic)

    # remove alpha channel  %some alpha=0 but RGB is not equal to [1., 1., 1.]
    img[img[:,:,3]==0] = np.ones((1,4))
    img = img[:,:,0:3]

    orig_img = np.append(orig_img, [img], axis=0)
    orig_type = np.append(orig_type, [poke_type[kk, :]], axis=0)

    kk += 1


for folder in os.listdir('data/Poke_viola/'):
    
    if not folder.startswith('.'):
        PkY = df_type.ix[folder,:].as_matrix()
        
        for pokemon in os.listdir('data/Poke_viola/' + folder):
            orig_type = np.append(orig_type, [PkY], axis=0)
            
            if not pokemon.startswith('.'):
                img = mpimg.imread('data/Poke_viola/' + folder + '/' + pokemon)
                
                if img.shape[-1] == 4:  # if transparent background, fill with white
                    img[:,:,:3] = img[:,:,:3] * (img[:,:,3:] > 0) + (img[:,:,3:] == 0 ) * 255
                    img = img[:,:,:3]
                
                img = resize(img, [40, 40, 3])

                im_list = im_list + [img]
        
orig_img = np.concatenate([orig_img, np.array(im_list)], 0)



# Construct discriminator and generator
lambda_DAE = 100
dist = 2  #tf.random_uniform([1], 2, 16, dtype = tf.int32)
shifted = tf.slice(Y, [0, 2], [batch_size, 18 - dist])
shifted2 = tf.slice(Y, [0,0],[batch_size, dist])
Y_wrong = tf.concat([shifted, shifted2], 1)

gen_sample, DAE_loss = generator(Z_plus_Y)
dis_real = discriminator(X, Y)
dis_fake = discriminator(gen_sample, Y)
dis_wrong = discriminator(X, Y_wrong)

# Define loss
dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=dis_real, labels=tf.ones_like(dis_real)))
dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=dis_fake, labels=tf.zeros_like(dis_fake)))
dis_loss_wrong = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=dis_wrong, labels=tf.zeros_like(dis_wrong)))


dis_loss = dis_loss_real + dis_loss_fake + dis_loss_wrong
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=dis_fake, labels=tf.ones_like(dis_fake)))

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'dis_' in var.name]
g_vars = [var for var in t_vars if 'gen_' in var.name]

# Define the L2 Regularization losses for discriminators and generators
dis_l2 = tf.reduce_sum([tf.reduce_sum(tf.square(i)) * 1e-1 for i in d_vars]) 
gen_l2 = tf.reduce_sum([tf.reduce_sum(tf.square(i)) * 1e-5 for i in g_vars])

dis_loss_raw = dis_loss
gen_loss_raw = gen_loss

dis_loss += dis_l2
gen_loss += gen_l2

# Optimizer for discriminator
var_dis = [weights[i] for i in weights if re.match('dis', i)]+[biases[i] for i in biases if re.match('dis', i)]
dis_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(dis_loss, var_list= var_dis + d_vars)

# Optimizer for generator parameters
var_gen = [weights[i] for i in weights if re.match('gen', i)]+[biases[i] for i in biases if re.match('gen', i)]
gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(gen_loss + lambda_DAE * DAE_loss, var_list= var_gen + g_vars)




# Initializing the variables
init = tf.global_variables_initializer()

# Create session and graph, initial variables
sess = tf.InteractiveSession()
sess.run(init)




# Load previous trained model and rewrite to variables, if exists
# Before run this, run the above first, to define variables and init it.
saver = tf.train.Saver()

try:
    saver.restore(sess, FLAGS.saved_model_path)

    print("Model restored.")

except:
    print("Model restore failed.")
    pass


total_batch = int(orig_img.shape[0]/batch_size)
print(len(orig_img))

# Training cycle
g_loss_train = 0
for epoch in range(training_epochs+1):

    # Loop over all batches
    # Save trained Variables every 100 epochs
    if epoch % 10 == 0 and epoch > 0:
        save_path = saver.save(sess, FLAGS.saved_model_path)

        # GENERATE Pokemon!!!
        PokeGAN = sess.run(
            gen_sample, feed_dict={Z: sample_Z(batch_size, gen_dim), Y: UserType})

        # Save generated pokemons to png file
        plotPoke(PokeGAN, 10, 5, caps, 
                 FLAGS.output_file_path+FLAGS.outfig_name_prefix+str(epoch)+'.png')

    
    for i in range(total_batch-1):

        index = np.random.choice(np.arange(len(orig_img)), size=batch_size)

        batch_xs = orig_img[index]
        batch_tp = orig_type[index]
        batch_zs = sample_Z(batch_size, gen_dim)
        
        # Run optimization op (backprop) and loss op (to get loss value)
        if g_loss_train < 1.5:
            _, d_loss_train = sess.run([dis_optimizer, dis_loss], feed_dict = {X: batch_xs, Z: batch_zs, Y: batch_tp})
        _, g_loss_train = sess.run([gen_optimizer, gen_loss], feed_dict = {Z: batch_zs, Y: batch_tp})
        

    # Display logs per epoch step
    if ((epoch == 0) or (epoch+1) % display_step == 0) or ((epoch+1) == training_epochs):
        print('Epoch: {0:04d}      Discriminator loss: {1:f}      Generator loss: {2:f}'.format(epoch+1, d_loss_train, g_loss_train))

print("Optimization Finished!")

