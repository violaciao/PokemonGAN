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
import datetime


time_now = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')


FLAGS = tf.app.flags.FLAGS


tf.flags.DEFINE_float("dropout_rate", 0.8, "dropout keep_prob rate")

tf.flags.DEFINE_string("outfig_name_prefix", "final_version_",
                       "Output plotting file's prefix.")
tf.flags.DEFINE_string("output_file_path", "generated_images/",
                       "Output images will be stored under this file path.")
tf.flags.DEFINE_string("saved_model_path", "saved_model_best/PokeGAN_model.ckpt",
                       "The trained model will be stored here.")
tf.flags.DEFINE_string("now", time_now, "Current time, which is used in the output file name.")



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
batch_size = 64
display_step = 5
examples_to_show = 9
kernel_dim = 5
n_kernels = 5


# Network Parameters
n_input = [40, 40, 3] # Pokemon data input (img shape: 40*40*3)
n_channel1 = 32
n_channel2 = 128 
n_channel3 = 256 
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
        
typeDuo = np.vstack({tuple(row) for row in typeDuo_list})

# Define the output plotting types and captions
# UserType = np.concatenate((np.eye(18), typeDuo[:46]), 0)
UserType = typeDuo[-64:]
caps = get_types(UserType)




# Define gen_sample
gen_sample, DAE_loss = generator(Z_plus_Y)



###############################################################
#            Generate Pokemons with Different Types           #
###############################################################

# Initializing the variables
init = tf.global_variables_initializer()

# Create session and graph, initial variables
sess = tf.InteractiveSession()
sess.run(init)




# Load previous trained model and rewrite to variables, if exists
# Before restoring the saved model, please run all the above codes first, to define variables and init it.
saver = tf.train.Saver()

try:
    saver.restore(sess, FLAGS.saved_model_path)

    print("Model restored.")

except:
    print("Model restore failed.")
    pass



# GENERATE Pokemon!!!

PokeGAN = sess.run(
    gen_sample, feed_dict={Z: sample_Z(batch_size, gen_dim), Y: UserType})

# Save generated pokemons to png file
plotPoke(PokeGAN, 5, 10, caps, 
         FLAGS.output_file_path+FLAGS.outfig_name_prefix+FLAGS.now+'.png')





