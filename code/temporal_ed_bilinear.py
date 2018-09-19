import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt
import random
from keras import backend as K


## prepare the pretrained VGG-16 model on ImageNet
## train the recurrent model end to end jointly

## prepare VGG network and loss
# import tensorflow.contrib.slim.nets as nets

# ----------------- sequence preprocessing ----------------- #
def input_seq_preprocess(inputs):
    ## convert batch of feature sequences to desired formats, np.float32
    ## inputs = 3D tensor with [batch, time, Channel]
    ## outputs = list of tensor, each of which [batch, 3*channel]
    
    # step1: unstack and cast to tf.float32
    inputs_list = tf.unstack(inputs, axis=1)
    inputs_list = [tf.cast(x, dtype=tf.float32) for x in inputs_list]
    
    outputs_stack = inputs_list   
   
    # step2: mirror sequence and reform input sequence by stacking
#    outputs_m = [inputs_list[0]]+inputs_list+[inputs_list[-1]] 
#    n_frames = len(outputs_m)

        # stack three images, the current time image is at the center 
#    outputs_stack = [tf.concat( [outputs_m[i-1], outputs_m[i], outputs_m[i+1]], axis=-1) 
#                    for i in range(1,n_frames-1)]


    return outputs_stack





def fully_connected_layers(x, num_units, scope, reuse):
    # the inputs is a tensor with shape [batch, width, height, channels]
    
    with tf.variable_scope(scope, 'fc', reuse=reuse) as sc:

        # inputs = tf.layers.batch_normalization(inputs, training=True)
        x = tf.layers.dense(x,num_units, activation=None)
        # if dropout_ratio != None:
        #     x = tf.layers.dropout(x, 1.0-dropout_ratio)
        
    return x





def tensor_product_local(X,W):
    #input: X in [batch, T, channel]
    #input: W in [T,T]
    #compute X^T * W * X  (* is multiplication)
    n_channels = X.shape[-1]
    A = K.dot(tf.transpose(X, [0,2,1]), W)
    # A_list = tf.unstack(A,axis=0)
    # X_list = tf.unstack(X,axis=0)
    # B_list = [tf.matmul(x,y) for x,y in zip(A_list,B_list)]
    # B = tf.stack(B_list,axis=0)
    B = K.batch_dot(A,X)
    return tf.reshape(B,[-1,n_channels*n_channels])


    
   
    
def tensor_product(inputs, time_conv_size, st_conv_filter_one):
    # input - [batch, time, channels]
    # output - [batch, time, channels^2]
    n_batches = inputs.shape[0]
    n_frames = inputs.shape[1]
    n_channels = inputs.shape[-1]

    local_size=time_conv_size
    

    x = tf.keras.layers.ZeroPadding1D((local_size//2))(inputs)
    W = tf.diag(st_conv_filter_one)

    y = [ tensor_product_local(x[:,i:i+local_size,:],W) for i in range(n_frames) ]

    outputs =tf.stack(y,axis=1) 

    return outputs





def tensor_conv_linear(inputs, time_conv_size, regularizer):
    # inputs is 3D with [batch_size, time, channel^2]
    # outputs is 3D with [batch_size, time, channel^2]
    
        
    st_conv_filter_one = tf.get_variable('st_conv_weights', shape=[time_conv_size],
                                     initializer=None,
                                     regularizer=regularizer)
    outputs = tensor_product(inputs, time_conv_size, st_conv_filter_one)

    return outputs






def tensor_pooling_linear(inputs, time_conv_size, regularizer):
    # inputs is 3D with [batch_size, time, channel^2]
    # outputs is 3D with [batch_size, time, channel^2]
    # Comparing with structure_tensor_conv, this function performs temporal avg pooling rather than conv
    x = tf.layers.average_pooling1d(inputs, [time_conv_size],strides=[1],
                                    padding='same')
    return x




def normalized_relu(x):
    # input: x - 3D tensor with [batch, time, channel^2]
    x_relu = tf.nn.relu(x)

    # Normalize by the highest activation
    max_values = tf.reduce_max(tf.abs(x_relu), 2, keepdims=True)+1e-5
    out = x_relu / max_values
    return out




def power_normalization(x):
    y = tf.sign(x) * tf.sqrt(tf.abs(x))
    return y





def act_fun_linear_tanh(x):
    #y = x * tf.tanh(x)
    #y=x*tf.sigmoid(x)
    #y = abs(x) * tf.tanh(x)
    #y = tf.atan(x)
    #y = tf.abs(x)
    lam = tf.get_variable('act_fun_weights',shape=[1],initializer=tf.initializers.ones)
    y = 2*lam*tf.sqrt(1+x**2/lam)-2*lam
    return y




def activation_fun(x, activation='relu'):

    if activation == 'relu':
        y = tf.nn.relu(x)
    elif activation == 'norm_relu':
        y = normalized_relu(x)
    elif activation == 'charbonnier':
        lam = tf.get_variable('act_fun_weights',shape=[1],initializer=tf.initializers.ones)
        y = 2*lam*tf.sqrt(1+x**2/lam)-2*lam
    elif activation == 'linear_tanh':
        y = x * tf.tanh(x)
    elif activation == 'swish':
        lam = tf.get_variable('act_fun_weights',shape=[1],initializer=tf.initializers.ones)
        y = x*tf.sigmoid(lam*x)
    else:
        print('[Error] activation function not valid')
        sys.exit()

    return y




def lp_normalization(x,p=2):
    
    norm_x = tf.maximum(tf.norm(x, ord=p, axis=-1, keepdims=True), 1e-6)
    
    return x/norm_x






# def temporal_encoder_decoder(x, regularizer, time_conv_size, n_nodes, is_training):
#     # x is the input, with [batch, time, n_dims]
#     norm_p = 4
#     with tf.variable_scope('temporal_pooling', reuse=tf.AUTO_REUSE):
        
#         # encoder1: frame-wise-tensor-product + st conv + 1x1 conv + act_fun + max_pooling
#         with tf.variable_scope('encoder1'):

#             x = tensor_conv_linear(x, time_conv_size,regularizer)
#             x = tf.nn.l2_normalize(x,axis=-1)
#             #x = lp_normalization(x,norm_p)
#             x = tf.layers.conv1d(x,160, kernel_size = [1], padding='SAME',
#                                  activation=None,
#                                  kernel_regularizer=regularizer)
            
#             x = tf.nn.dropout(x, 0.5)
#             x = tf.nn.relu(x)
            
#             x = tf.layers.max_pooling1d(x, [2], strides=2, padding='SAME')

#         # encoder2: frame-wise-tensor-product + 1x1 conv + t-conv + act_fun + max_pooling
#         with tf.variable_scope('encoder2'):
#             x = tensor_conv_linear(x, time_conv_size,regularizer)
#             x = tf.nn.l2_normalize(x,axis=-1)
#             #x = lp_normalization(x,norm_p)
#             x = tf.layers.conv1d(x,192, kernel_size = [1], padding='SAME',
#                                  activation=None,
#                                  kernel_regularizer=regularizer)
            
#             x = tf.nn.dropout(x, 0.5)
#             x = tf.nn.relu(x)

#             x = tf.layers.max_pooling1d(x, [2], strides=2, padding='SAME')

       


#         # decoder1: frame-wise-tensor-product + 1x1 conv + t-conv + act_fun + max_pooling
#         with tf.variable_scope('decoder1'):
#             x = tf.keras.layers.UpSampling1D(size=(2))(x)
#             x = tensor_conv_linear(x, time_conv_size,regularizer)
#             x = tf.nn.l2_normalize(x,axis=-1)
#             #x = lp_normalization(x,norm_p)
#             x = tf.layers.conv1d(x,160, kernel_size = [1], padding='SAME',
#                                  activation=None,
#                                  kernel_regularizer=regularizer)
            
#             x = tf.nn.dropout(x, 0.5)
#             x = tf.nn.relu(x)

#         # decoder2: frame-wise-tensor-product + 1x1 conv + t-conv + act_fun + max_pooling
#         with tf.variable_scope('decoder2'):
#             x = tf.keras.layers.UpSampling1D(size=(2))(x)
#             x = tensor_conv_linear(x, time_conv_size,regularizer)
#             x = tf.nn.l2_normalize(x,axis=-1)
#             #x = lp_normalization(x,norm_p)
#             x = tf.layers.conv1d(x,128, kernel_size = [1], padding='SAME',
#                                  activation=None,
#                                  kernel_regularizer=regularizer)
            
#             x = tf.nn.dropout(x, 0.5)
#             x = tf.nn.relu(x)

#         return tf.unstack(x,axis=1)
    
    





def temporal_encoder_decoder(x, regularizer, time_conv_size, n_nodes, activation='relu',
                            is_training=False):
    # x is the input, with [batch, time, n_dims]
    norm_p = 4
    with tf.variable_scope('temporal_pooling', reuse=tf.AUTO_REUSE):
        
        ## encoders
        for i in range(len(n_nodes)):
        
            with tf.variable_scope('encoder{:d}'.format(i+1)):

                # x = tensor_conv_linear(x, time_conv_size,regularizer)
                # x = tf.nn.l2_normalize(x,axis=-1)
                #x = lp_normalization(x,norm_p)
                x = tf.layers.conv1d(x,n_nodes[i], kernel_size = time_conv_size, padding='SAME',
                                     activation=None,
                                     kernel_regularizer=regularizer)
                
                x = tf.nn.dropout(x, 0.7)
                x = activation_fun(x,activation)
                
                x = tf.layers.max_pooling1d(x, [2], strides=2, padding='SAME')




        # decoder1: frame-wise-tensor-product + 1x1 conv + t-conv + act_fun + max_pooling
        for i in range(len(n_nodes)):
            with tf.variable_scope('decoder{:d}'.format(i+1)):
                x = tf.keras.layers.UpSampling1D(size=(2))(x)
                # x = tensor_conv_linear(x, time_conv_size,regularizer)
                # x = tf.nn.l2_normalize(x,axis=-1)
                #x = lp_normalization(x,norm_p)
                x = tf.layers.conv1d(x,n_nodes[-i-1], kernel_size = time_conv_size, padding='SAME',
                                     activation=None,
                                     kernel_regularizer=regularizer)
                
                x = tf.nn.dropout(x, 0.7)
                x = activation_fun(x,activation)


        return tf.unstack(x,axis=1)




def end_to_end_tensor_flow(inputs, model_options):
    # inputs = 3D tensor with [batch, time, channel]
    # outputs = 3D tensor with [batch, time, n_classes]
    

    n_batches=inputs.shape[0]
    n_frames = inputs.shape[1]
    reuse = model_options['reuse']
    scope = model_options['scope']
    output_dims = model_options['output_dims']
    dropout_keep_prob = model_options['dropout_keep_prob']
    regularizer =  model_options['regularizer']
    is_training = model_options['is_training']
    time_conv_size = model_options['time_conv_size']
    n_nodes = model_options['n_nodes']
    act_type = model_options['activation']
    with tf.variable_scope(scope, initializer=tf.contrib.layers.xavier_initializer()) as sc:

        # split frames along the time dimension
        inputs_list = input_seq_preprocess(inputs)
       
        # temporal encoder-decoder layer
        logits = tf.stack(inputs_list,axis=1)
        logits = temporal_encoder_decoder(logits, regularizer, time_conv_size, n_nodes, act_type,
                                          is_training)
        
        # fully connected layer for each frame
        logits_list_fusion = [fully_connected_layers(x, output_dims, sc, tf.AUTO_REUSE)
                                for x in logits] 
        logits_sm_fusion = [tf.nn.softmax(x) for x in logits_list_fusion]

        if model_options['is_training']:
            return tf.stack(logits_list_fusion, axis=1)
        else:
            return tf.stack(logits_sm_fusion, axis=1)



