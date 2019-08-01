import numpy as np
import keras
import sys
from keras.models import Sequential, Model
from keras.engine.topology import Layer
from keras.layers import Input, Dense, TimeDistributed, merge, Lambda
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
from keras import backend as K

import scipy


from keras.activations import relu
from functools import partial
clipped_relu = partial(relu, max_value=5)


def max_filter(x):
    # Max over the best filter score (like ICRA paper)
    max_values = K.max(x, 2, keepdims=True)
    max_flag = tf.greater_equal(x, max_values)
    out = x * tf.cast(max_flag, tf.float32)
    return out

def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True)+1e-5
    out = x / max_values
    return out

def WaveNet_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)  
    return keras.layers.Multiply()([tanh_out, sigm_out])



def lp_normalization(x,p=2):
    if p == 2:
        return K.l2_normalize(x, axis=-1)
    else:
        norm_x = tf.maximum(tf.norm(x, ord=p, axis=-1, keepdims=True), 1e-6)
        return x/norm_x


def power_normalization(x):
    y = tf.sign(x) * tf.sqrt(tf.abs(x))
    return y


def is_power2(x):
    return x!=0 and ((x & (x-1))==0)












class SqrtAcfun(Layer):
    def __init__(self, theta=1e-3,**kwargs):
        self.theta = theta
        super(SqrtAcfun, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        self.gamma = self.add_weight(name='gamma',shape=[1],
                                     initializer=keras.initializers.Constant(value=self.theta),
                                     trainable=True)
        super(SqrtAcfun,self).build(input_shape)

    def call(self,x):
        x = K.sign(x)* (K.sqrt(K.abs(x)+self.gamma)-K.sqrt(self.gamma))

        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape)






class QRDecompose(Layer):
    def __init__(self, **kwargs):
        super(QRDecompose, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(QRDecompose,self).build(input_shape)

    def call(self,x):

        q, r = tf.linalg.qr(x)
        return q
    
    def compute_output_shape(self, input_shape):
        return (input_shape)






### the output and the input have the same dimension
class TensorChainDecomposePooling(Layer):
    def __init__(self, n_recur=1, same_mat=False, out_fusion_type='mean',
                 act_fun_in='linear', act_fun_out='linear', 
                 stride=2, trainable=True, 
                 **kwargs):
        
        
        self.out_fusion_type=out_fusion_type
        self.n_recur = n_recur
        self.stride = stride
        self.trainable = trainable
        self.same_mat=same_mat

        if act_fun_in=='linear':
            self.act_fun_in = keras.activations.linear
        elif act_fun_in=='tanh':
            self.act_fun_in = K.tanh
        else:
            print('[ERROR]: no such activation function for input. Program terminates')
            sys.exit()


        if act_fun_out=='linear':
            self.act_fun_out = keras.activations.linear
        elif act_fun_out=='tanh':
            self.act_fun_out = K.tanh
        else:
            print('[ERROR]: no such activation function for output. Program terminates')
            sys.exit()


        super(TensorChainDecomposePooling, self).__init__(**kwargs)


    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]

        # self.out_dim = in_dim
        self.core_dim = in_dim

        if not self.same_mat:
            self.factorInMats = []
            self.factorOutMats = []
            self.factorInBias = []
            self.factorOutBias = []
            self.core_diag_list = []

            for ii in range(self.n_recur):
                self.factorInMats.append(self.add_weight(name='mat_in{:d}'.format(ii), 
                                    shape=[in_dim, in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)
                                    )
                self.factorInBias.append(self.add_weight(name='b_in{:d}'.format(ii), 
                                    shape=[in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)
                                    )

                self.factorOutMats.append(self.add_weight(name='mat_out{:d}'.format(ii), 
                                    shape=[in_dim, in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)
                                    )
                self.factorOutBias.append(self.add_weight(name='b_out{:d}'.format(ii), 
                                    shape=[in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)
                                    )           
        else:
            self.factorInMats = []
            self.factorOutMats = []
            self.factorInBias = []
            self.factorOutBias = []
            self.core_diag_list = []


            W_in = self.add_weight(name='mat_in', 
                                    shape=[in_dim, in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)

            b_in = self.add_weight(name='b_in', 
                                    shape=[in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)

            W_out = self.add_weight(name='mat_out', 
                                    shape=[in_dim, in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)

            b_out = self.add_weight(name='b_out', 
                                    shape=[in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)


            for ii in range(self.n_recur):
                self.factorInMats.append(W_in)
                self.factorInBias.append(b_in)
                self.factorOutMats.append(W_out)
                self.factorOutBias.append(b_out)           

        for ii in range(self.n_recur+1):
            if not self.trainable:        
                self.core_diag_list.append(self.add_weight(name='core_diag_{}'.format(ii), 
                                            shape=[self.core_dim], 
                                            initializer=keras.initializers.Constant(value=1.0),
                                            trainable=False)
                                          )
            else:
                self.core_diag_list.append(self.add_weight(name='core_diag_{}'.format(ii), 
                                            shape=[self.core_dim], 
                                            initializer='glorot_normal',
                                            trainable=True)
                                          )


        super(TensorChainDecomposePooling, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        n_frames = X.get_shape().as_list()[1]
        z_list = []

        ## before iteration, we do the first bilinear fusion
        lambda_0 = K.reshape(K.abs(self.core_diag_list[0]), [1, 1, -1])
        z = lambda_0 * X * X
        z = self.act_fun_out(z)
        z_list.append(z)

        for ii in range(self.n_recur):

            ## (1) obtain x_{t+1} = U_tx_t + b_t 
            X = K.dot(X, self.factorInMats[ii]) + self.factorInBias[ii]
            X = self.act_fun_in(X)
            lambda_i = K.reshape(K.abs(self.core_diag_list[ii+1]), [1, 1, -1])
            z = lambda_i * X * X
            zout = K.dot(z, self.factorOutMats[ii]) + self.factorOutBias[ii]
            zout = self.act_fun_out(zout)
            z_list.append(zout)

        if self.out_fusion_type == 'mean':
            ## compute the mean value 
            z_list_tensor = K.stack(z_list, axis=-2)
            out = K.mean(z_list_tensor, axis=-2)
        
        elif self.out_fusion_type == 'concate':
            ## concatente features
            out = K.concatenate(z_list, axis=-1)

        else:
            print('[ERROR] no such fusion method. Program terminates')
            sys.exit()


        self.out_dim = out.get_shape().as_list()[-1]
        ### now out is [batch, T, out_dim], we do temporal local pooling
        #### zero padding
        # out = ZeroPadding1D((self.time_window_size//2))(out)
        # W = tf.reshape(self.conv_filter, [1, -1, 1]) # [1, |Nt|, 1]
        # out_pool_list = [ K.sum(out[:, i:i+self.time_window_size, :]*W, axis=1)
        #                  for i in range(0,n_frames, self.stride) ]

        # out_pool = K.stack(out_pool_list,axis=1)
        out_pool = AveragePooling1D(self.stride)(out)

        return out_pool


    def get_weights(self):
        return self.factorInMats+self.factorOutMats


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)












### since CP decomposition is not unique, we think of more than one types of decompositions.
### the output and the input have the same dimension
class TensorStarDecomposePooling(Layer):
    def __init__(self, n_recur=1, 
                 same_mat=False, 
                 use_bias=True,
                 out_fusion_type='sum',
                 act_fun_in='linear', 
                 act_fun_out='linear',
                 stride=2, 
                 trainable=True, 
                 **kwargs):
        
        
        self.out_fusion_type=out_fusion_type
        self.n_recur = n_recur
        self.stride = stride
        self.trainable = trainable
        self.same_mat=same_mat
        self.use_bias = use_bias

        if act_fun_in=='linear':
            self.act_fun_in = keras.activations.linear
        elif act_fun_in=='tanh':
            self.act_fun_in = K.tanh
        else:
            print('[ERROR]: no such activation function for input. Program terminates')
            sys.exit()


        if act_fun_out=='linear':
            self.act_fun_out = keras.activations.linear
        elif act_fun_out=='tanh':
            self.act_fun_out = K.tanh
        else:
            print('[ERROR]: no such activation function for output. Program terminates')
            sys.exit()





        super(TensorStarDecomposePooling, self).__init__(**kwargs)


    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]

        # self.out_dim = in_dim
        self.core_dim = in_dim

        # if self.out_dim != self.core_dim or in_dim != self.core_dim:
        #     print('[ERROR]: when reduce_dim=True, keep in_dim == out_dim == core_dim!')
        #     sys.exit()                


        if not self.same_mat:
            self.factorInMats = []
            self.factorOutMats = []

            if self.use_bias:
                self.factorInBias = []
                self.factorOutBias = []

            self.core_diag_list = []

            for ii in range(self.n_recur):
                self.factorInMats.append(self.add_weight(name='mat_in{:d}'.format(ii), 
                                    shape=[in_dim, in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)
                                    )

                if self.use_bias:
                    self.factorInBias.append(self.add_weight(name='b_in{:d}'.format(ii), 
                                        shape=[in_dim], 
                                        initializer='glorot_normal',
                                        trainable=True)
                                        )

                self.factorOutMats.append(self.add_weight(name='mat_out{:d}'.format(ii), 
                                    shape=[in_dim, in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)
                                    )
                if self.use_bias:
                    self.factorOutBias.append(self.add_weight(name='b_out{:d}'.format(ii), 
                                        shape=[in_dim], 
                                        initializer='glorot_normal',
                                        trainable=True)
                                        )           
        else:
            self.factorInMats = []
            self.factorOutMats = []
            self.factorInBias = []
            self.factorOutBias = []
            self.core_diag_list = []


            W_in = self.add_weight(name='mat_in', 
                                    shape=[in_dim, in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)

            if self.use_bias:
                b_in = self.add_weight(name='b_in', 
                                        shape=[in_dim], 
                                        initializer='glorot_normal',
                                        trainable=True)

            W_out = self.add_weight(name='mat_out', 
                                    shape=[in_dim, in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)

            if self.use_bias:
                b_out = self.add_weight(name='b_out', 
                                        shape=[in_dim], 
                                        initializer='glorot_normal',
                                        trainable=True)


            for ii in range(self.n_recur):
                self.factorInMats.append(W_in)
                self.factorOutMats.append(W_out)

                if self.use_bias:
                    self.factorInBias.append(b_in)
                    self.factorOutBias.append(b_out)           


        for ii in range(self.n_recur+1):
            if not self.trainable:        
                self.core_diag_list.append(self.add_weight(name='core_diag_{}'.format(ii), 
                                            shape=[self.core_dim], 
                                            initializer=keras.initializers.Constant(value=1.0),
                                            trainable=False)
                                          )
            else:
                self.core_diag_list.append(self.add_weight(name='core_diag_{}'.format(ii), 
                                            shape=[self.core_dim], 
                                            initializer='glorot_normal',
                                            trainable=True)
                                          )

        super(TensorStarDecomposePooling, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        n_frames = X.get_shape().as_list()[1]
        z_list = []

        ## before iteration, we do the first bilinear fusion
        lambda_0 = K.reshape(K.abs(self.core_diag_list[0]), [1, 1, -1])
        z = lambda_0 * X * X
        z = self.act_fun_out(z)
        z_list.append(z)

        for ii in range(self.n_recur):
            ## (1) obtain x_{t+1} = U_tx_t + b_t 
            # U = QRDecompose()(self.factorInMats[ii])
            U = self.factorInMats[ii]
            if self.use_bias:
                X1 = K.dot(X, U) + self.factorInBias[ii]
            else:
                X1 = K.dot(X, U)

            X1 = self.act_fun_in(X1)
            lambda_i = K.reshape(K.abs(self.core_diag_list[ii+1]), [1, 1, -1])
            z = lambda_i * X1 * X1

            # W = QRDecompose()(self.factorOutMats[ii])
            W = self.factorOutMats[ii]

            if self.use_bias:
                zout = K.dot(z, W) + self.factorOutBias[ii]
            else:
                zout = K.dot(z, W)

            zout = self.act_fun_out(zout)
            z_list.append(zout)

        if self.out_fusion_type == 'sum':
            # ## compute the mean value 
            z_list_tensor = K.stack(z_list, axis=-1)
            out = K.sum(z_list_tensor, axis=-1)

        elif self.out_fusion_type == 'concate':
            ## concatente features
            out = K.concatenate(z_list, axis=-1)
            
        self.out_dim = out.get_shape().as_list()[-1]

        ## now out is [batch, T, out_dim], we do temporal local pooling
        ### zero padding
        # out = ZeroPadding1D((self.time_window_size//2))(out)
        # W = tf.reshape(self.conv_filter, [1, -1, 1]) # [1, |Nt|, 1]
        # out_pool_list = [ K.sum(out[:, i:i+self.time_window_size, :]*W, axis=1)
        #                  for i in range(0,n_frames, self.stride)  ]

        # out_pool = K.stack(out_pool_list,axis=1)
        out_pool = AveragePooling1D(self.stride)(out) 
        # out_pool = MaxPooling1D(self.stride)(out) 

        return out_pool


    def get_weights(self):
        return self.factorInMats+self.factorOutMats


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)







### Recursive CP tensor decomposition
class TensorDecomposePooling(Layer):
    def __init__(self, out_dim, core_dim,
                 time_window_size, stride=2, trainable=False, 
                 reduce_dim=False, n_recur=1,
                 **kwargs):
        
        self.out_dim = out_dim
        self.core_dim = core_dim
        
        self.reduce_dim = reduce_dim
        self.n_recur = n_recur

        self.time_window_size = time_window_size
        self.stride = stride
        self.trainable = trainable

        # self.act_fun = keras.activations.linear
        self.act_fun = K.tanh
        super(TensorDecomposePooling, self).__init__(**kwargs)


    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]

        if self.reduce_dim:
            if is_power2(in_dim) and is_power2(self.out_dim) and is_power2(self.core_dim):
                self.in_recur = int(np.log2(in_dim // self.core_dim))
                self.out_recur = int(np.log2(self.out_dim // self.core_dim) )

                print('-- # inner recursion={}'.format(self.in_recur))
                print('-- # outer recursion={}'.format(self.out_recur))

                if self.core_dim < 8:
                    print('[ERROR]: core_dim={}. Too small! Program terminates'.format(self.core_dim))
                    sys.exit()
            else:
                print('[ERROR]: the input feature dim is required to be power of 2.')
                sys.exit()

        else:
            self.in_recur=self.n_recur
            self.out_recur=self.n_recur
            if self.out_dim != self.core_dim or in_dim != self.core_dim:
                print('[ERROR]: when reduce_dim=True, keep in_dim == out_dim == core_dim!')
                sys.exit()                


        if not self.trainable:
            
            self.core_diag = self.add_weight(name='core_diag', 
                                shape=[self.core_dim], 
                                initializer=keras.initializers.Constant(value=1.0),
                                trainable=False)
        else:
            self.core_diag = self.add_weight(name='core_diag', 
                                shape=[self.core_dim], 
                                initializer='glorot_normal',
                                trainable=True)


        # self.conv_filter = self.add_weight(name='conv_kernel', 
        #                             shape=[self.time_window_size], 
        #                             initializer=keras.initializers.Constant(value=1.0/self.time_window_size),
        #                             trainable=False)

        self.conv_filter = self.add_weight(name='conv_kernel', 
                                shape=[self.time_window_size], 
                                initializer='glorot_normal',
                                trainable=True) 


        self.factorInMats = []
        self.factorOutMats = []
        self.factorInBias = []
        self.factorOutBias = []

        if self.reduce_dim:
            for ii in range(self.in_recur):
                self.factorInMats.append(self.add_weight(name='mat_in{:d}'.format(ii), 
                                    shape=[in_dim // 2**(ii), in_dim//2**(ii+1)], 
                                    initializer='glorot_normal',
                                    trainable=True)
                                    )
                self.factorInBias.append(self.add_weight(name='b_in{:d}'.format(ii), 
                                    shape=[in_dim//2**(ii+1)], 
                                    initializer=keras.initializers.Constant(value=0.0),
                                    trainable=True)
                                    )

            for ii in range(self.out_recur):
                self.factorOutMats.append(self.add_weight(name='mat_out{:d}'.format(ii), 
                                    shape=[self.out_dim // 2**(ii+1), self.out_dim//2**(ii)], 
                                    initializer='glorot_normal',
                                    trainable=True)
                                    )
                self.factorOutBias.append(self.add_weight(name='b_out{:d}'.format(ii), 
                                    shape=[self.out_dim//2**(ii)], 
                                    initializer=keras.initializers.Constant(value=0.0),
                                    trainable=True)
                                    )           
        else:
            for ii in range(self.in_recur):
                self.factorInMats.append(self.add_weight(name='mat_in{:d}'.format(ii), 
                                    shape=[in_dim, in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)
                                    )
                self.factorInBias.append(self.add_weight(name='b_in{:d}'.format(ii), 
                                    shape=[in_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)
                                    )

            for ii in range(self.out_recur):
                self.factorOutMats.append(self.add_weight(name='mat_out{:d}'.format(ii), 
                                    shape=[self.out_dim, self.out_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)
                                    )
                self.factorOutBias.append(self.add_weight(name='b_out{:d}'.format(ii), 
                                    shape=[self.out_dim], 
                                    initializer='glorot_normal',
                                    trainable=True)
                                    )           

        super(TensorDecomposePooling, self).build(input_shape)




    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        n_frames = X.get_shape().as_list()[1]
        out = X
        for ii in range(self.in_recur):
            out = K.dot(out, self.factorInMats[ii]) + self.factorInBias[ii]
            out = self.act_fun(out)

        ### now we reach the core tensor
        ### do elementwise multiplication, self.core_diag is broadcasted.
        out = K.reshape(K.abs(self.core_diag), [1, 1, -1])*out*out

        ### now we go to the output dim
        for ii in range(self.out_recur):
            
            out = K.dot(out, self.factorOutMats[-ii-1]) + self.factorOutBias[-ii-1]
            out = self.act_fun(out)


        ### now out is [batch, T, out_dim], we do temporal local pooling
        #### zero padding
        out = ZeroPadding1D((self.time_window_size//2))(out)

        W = tf.reshape(self.conv_filter, [1, -1, 1]) # [1, |Nt|, 1]
        out_pool_list = [ K.sum(out[:, i:i+self.time_window_size, :]*W, axis=1)
                         for i in range(0,n_frames, self.stride)  ]

        out_pool = K.stack(out_pool_list,axis=1)

        return out_pool


    def get_weights(self):
        return self.factorInMats+self.factorOutMats


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)







from keras.constraints import Constraint

class non_neg_unit_norm (Constraint):
    def __init__(self, axis=0):
        self.axis=axis

    def __call__(self, w):
        w *= K.cast(K.greater_equal(w, 0.), K.floatx()) # non negative constraint
        
        #w = w / (K.epsilon() + K.sqrt(K.sum(K.square(w),
        #                                       axis=self.axis,
        #                                       keepdims=True)))
        w = w / (K.epsilon() + K.abs(w))
        #w *= K.cast(K.greater_equal(w, 0.), K.floatx()) # non negative constraint
        

        return w


class hard_binary_constraint (Constraint):
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def __call__(self, w):
        # we assume that w is initialized by glorot_normal,
        sigma = np.sqrt(2 / (self.n_rows+self.n_cols))
        w = w / (K.epsilon() + K.abs(w)) # the number of 2*sigma refers to https://www.tensorflow.org/api_docs/python/tf/random/truncated_normal
        # w = w * (2*sigma)/K.max(K.abs(w))
        # w = K.tanh(2*w)
        # w = K.sign(w) * (K.abs(w)**0.5)
        # w = w * sigma*2e-3
        return w


class tanh_binary_constraint (Constraint):
    def __init__(self, tanh_k):
        self.tanh_k = tanh_k

    def __call__(self, w):
        # we assume that w is initialized by glorot_normal,
        
        w = K.tanh(self.tanh_k*w)
        return w


class TensorRelaxationPooling(Layer):
    def __init__(self,
                 n_basis,
                 use_bias=False,
                 use_normalization=False,
                 constraint_type='tanh',
                 tanh_k = 1.5,
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        self.n_basis = n_basis
        self.out_dim = n_basis**2
        self.out_fusion_type = out_fusion_type
        self.stride = stride
        self.use_bias = use_bias
        self.use_normalization = use_normalization
        self.time_window_size = time_window_size
        self.activation = 'linear'
        self.constraint_type = constraint_type
        self.tanh_k = tanh_k

        super(TensorRelaxationPooling, self).__init__(**kwargs)



    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]
        if self.n_basis > in_dim:
           print('[ERROR]: n_basis must not be larger than in_dim! Program terminates')
           sys.exit()

        scalar_learnable=True

        if self.constraint_type == 'binary':
            weight_constraint = hard_binary_constraint(in_dim, self.n_basis)
            # scalar_learnable=False

        elif self.constraint_type == 'tanh_k2':
            weight_constraint = tanh_binary_constraint(2)

        elif self.constraint_type == 'tanh_k1':
            weight_constraint = tanh_binary_constraint(1.0+1e-2)
        elif self.constraint_type == 'tanh_k1.5':
            weight_constraint = tanh_binary_constraint(1.5)
        elif self.constraint_type == 'tanh_k2.5':
            weight_constraint = tanh_binary_constraint(2.5)
        elif self.constraint_type == 0:
            weight_constraint=None
            scalar_learnable=False
        else:
            print('[ERROR]: no such weight constraint.')
            sys.exit()





        ## define the two matrix with orthogonal columns
        stddev = np.sqrt(2 / (in_dim+self.n_basis))
        weight_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=stddev, seed=None)



        self.E = self.add_weight(name='E',
                            shape=[in_dim, self.n_basis], 
                            # initializer=keras.initializers.Orthogonal(),
                            # initializer='ones',
                            initializer=weight_initializer,
                            constraint=weight_constraint,
                            # constraint=non_neg_unit_norm(axis=0),
                            trainable=True)


        self.F = self.add_weight(name='F',
                            shape=[in_dim, self.n_basis], 
                            # initializer=keras.initializers.Orthogonal(),
                            # initializer='ones',
                            initializer=weight_initializer,
                            constraint=weight_constraint,
                            trainable=True)




        self.G = self.add_weight(name='G',
                            shape=[1], 
                            # initializer=keras.initializers.Orthogonal(),
                            # initializer='ones',
                            initializer=keras.initializers.Constant(1),
                            constraint=keras.constraints.NonNeg(),
                            trainable=scalar_learnable)



        if self.use_bias:
            self.bx = self.add_weight(name='bias_x',
                            shape=[self.n_basis], 
                            initializer='glorot_normal',
                            trainable=True)

            self.by = self.add_weight(name='bias_y',
                            shape=[self.n_basis], 
                            initializer='glorot_normal',
                            trainable=True)
        else:
            self.bx = self.add_weight(name='bias_x',
                            shape=[self.n_basis], 
                            initializer='zeros',
                            trainable=False)

            self.by = self.add_weight(name='bias_y',
                            shape=[self.n_basis], 
                            initializer='zeros',
                            trainable=False)


        if self.out_fusion_type == 'w-sum':
            self.conv_filter = self.add_weight(name='conv_kernel', 
                        shape=[self.time_window_size], 
                        initializer='glorot_normal',
                        trainable=True) 


        super(TensorRelaxationPooling, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        n_frames = X.get_shape().as_list()[1]
        # self.E = K.l2_normalize(self.E, axis=0)
        # self.F = K.l2_normalize(self.F, axis=0)
        # self.E_softmax = K.softmax(self.E, axis=0)
        # self.F_softmax = K.softmax(self.F, axis=0)
        
        # apply uni-norm, non-negative constraint
        #self.E *= K.cast(K.greater_equal(self.E, 0), float) # non negative constraint
        #
        #self.E = self.E / (K.epsilon() + K.sqrt(K.sum(K.square(self.E),
        #                                       axis=0,
        #                                       keepdims=True)))
        #self.F *= K.cast(K.greater_equal(self.F, 0.), float) # non negative constraint
        #
        #self.F = self.F / (K.epsilon() + K.sqrt(K.sum(K.square(self.F),
        #                                       axis=0,
        #                                       keepdims=True)))


        z1 = K.dot(X, self.E  )+K.reshape(self.bx, [1,1,-1])
        z2 = K.dot(X, self.F  )+K.reshape(self.by, [1,1,-1])

        if self.activation == 'tanh':
            z1 = K.tanh(z1)
            z2 = K.tanh(z2)


        # outer product
        z1 = K.expand_dims(z1, axis=-1)
        z2 = K.expand_dims(z2, axis=-2)
        z = tf.matmul(z1, z2)
        z = self.G*K.reshape(z, [-1, n_frames, self.n_basis**2])
        
       

        ## pooling to reduce time sequence
        if self.out_fusion_type == 'avg':
            out_pool = AveragePooling1D(pool_size=self.stride,
                                        strides=self.stride,
                                        padding='same')(z) 
        elif self.out_fusion_type == 'max':
            out_pool = MaxPooling1D(self.stride)(z) 
        elif self.out_fusion_type == 'w-sum':
            ### now out is [batch, T, out_dim], we do temporal local pooling
            #### zero padding
            out = ZeroPadding1D((self.time_window_size//2))(z)

            W = tf.reshape(self.conv_filter, [1, -1, 1]) # [1, |Nt|, 1]
            out_pool_list = [ K.sum(out[:, i:i+self.time_window_size, :]*W, axis=1)
                             for i in range(0,n_frames, self.stride)  ]

            out_pool = K.stack(out_pool_list,axis=1)
        elif self.out_fusion_type == 'linearproj':
            out_pool = Conv1D(self.out_dim, 1, strides=self.stride, padding='same')(z)


        return out_pool


    def get_mat(self):
        return [self.E, self.F]


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)














class RPBinaryPooling(Layer):
    def __init__(self,
                 n_basis=8,
                 n_components=1,
                 use_normalization=False,
                 activation=None,
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        self.n_basis = n_basis
        self.out_dim = n_basis**2
        self.n_components=n_components
        self.out_fusion_type = out_fusion_type
        self.stride = stride
        self.use_normalization = use_normalization
        self.time_window_size = time_window_size

        if activation == None:
            self.act_fun = tf.keras.activations.linear
        elif activation == 'tanh':
            self.act_fun = tf.keras.activations.tanh
        elif activation == 'relu':
            self.act_fun = tf.keras.activations.relu

        super(RPBinaryPooling, self).__init__(**kwargs)



    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]

        if self.n_basis > in_dim:
           print('[ERROR]: n_basis must not be larger than in_dim! Program terminates')
           sys.exit()



        ## define the two matrix with orthogonal columns
        self.E_list = []
        self.F_list = []
        self.we_list = []
        for n in range(self.n_components):
            E0 = np.sign(np.random.standard_normal([in_dim, in_dim]))
            # E0_signchange = np.sum(((np.roll(E0,1,axis=0)-E0) != 0).astype(int), axis=0)
            # E0 = E0[:, np.argsort(E0_signchange)]
            E0 = E0[:, :self.n_basis]

            self.E_list.append( self.add_weight(name='E_{}'.format(n),
                                                shape=[in_dim, self.n_basis], 
                                                initializer=keras.initializers.Constant(E0),
                                                trainable=False)
                               )
            self.we_list.append(self.add_weight(name='we_{}'.format(n),
                                                shape=[1], 
                                                initializer=keras.initializers.Constant(1.0/self.n_components),
                                                trainable=True))


            F0 = np.sign(np.random.standard_normal([in_dim, in_dim]))
            # F0_signchange = np.sum(((np.roll(F0,1,axis=0)-F0) != 0).astype(int), axis=0)
            # F0 = F0[:, np.argsort(F0_signchange)]
            F0 = F0[:, :self.n_basis]

            self.F_list.append( self.add_weight(name='F_{}'.format(n),
                                                shape=[in_dim, self.n_basis], 
                                                initializer=keras.initializers.Constant(F0),
                                                trainable=False)
                               )


        super(RPBinaryPooling, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        n_frames = X.get_shape().as_list()[1]
        in_dim = float(X.get_shape().as_list()[-1])
        z = 0.0

        for ii in range(self.n_components):

            z1 = K.dot(X, self.E_list[ii]  )
            z2 = K.dot(X, self.F_list[ii]  )

            # outer product
            z1 = self.act_fun(K.expand_dims(z1, axis=-1)) 
            z2 = self.act_fun(K.expand_dims(z2, axis=-2)) 
            z12 = tf.matmul(z1, z2)
            z12 = K.reshape(z12, [-1, n_frames, self.n_basis**2]) / in_dim

            ## use power and l2 normalization
            if self.use_normalization:
                z12 = K.sign(z12) * K.sqrt(K.abs(z12))
                z12 = K.l2_normalize(z12, axis=-1)

            z += z12 * self.we_list[ii]
        
        ## pooling to reduce time sequence
        if self.out_fusion_type == 'avg':
            out_pool = AveragePooling1D(pool_size=self.stride,
                                        strides=self.stride,
                                        padding='same')(z) 
        elif self.out_fusion_type == 'max':
            out_pool = MaxPooling1D(self.stride)(z) 

        return out_pool


    def get_mat(self):
        return [self.E_list, self.F_list]


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)







class RPBinaryPooling2(Layer):
    def __init__(self,
                 n_basis=8,
                 n_components=1,
                 use_normalization=False,
                 activation=None,
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        self.n_basis = n_basis
        self.out_dim = n_basis**2
        self.n_components=n_components
        self.out_fusion_type = out_fusion_type
        self.stride = stride
        self.use_normalization = use_normalization
        self.time_window_size = time_window_size

        if activation == None:
            self.act_fun = tf.keras.activations.linear
        elif activation == 'tanh':
            self.act_fun = tf.keras.activations.tanh
        elif activation == 'relu':
            self.act_fun = tf.keras.activations.relu

        super(RPBinaryPooling2, self).__init__(**kwargs)



    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]

        if self.n_basis > in_dim:
           print('[ERROR]: n_basis must not be larger than in_dim! Program terminates')
           sys.exit()



        ## define the two matrix with orthogonal columns
        self.E_list = []
        self.F_list = []
        for n in range(self.n_components):
            E0 = np.sign(np.random.standard_normal([in_dim, in_dim]))
            # E0_signchange = np.sum(((np.roll(E0,1,axis=0)-E0) != 0).astype(int), axis=0)
            # E0 = E0[:, np.argsort(E0_signchange)]
            E0 = E0[:, :self.n_basis]

            self.E_list.append( self.add_weight(name='E_{}'.format(n),
                                                shape=[in_dim, self.n_basis], 
                                                initializer=keras.initializers.Constant(E0),
                                                trainable=False)
                               )

            F0 = np.sign(np.random.standard_normal([in_dim, in_dim]))
            # F0_signchange = np.sum(((np.roll(F0,1,axis=0)-F0) != 0).astype(int), axis=0)
            # F0 = F0[:, np.argsort(F0_signchange)]
            F0 = F0[:, :self.n_basis]

            self.F_list.append( self.add_weight(name='F_{}'.format(n),
                                                shape=[in_dim, self.n_basis], 
                                                initializer=keras.initializers.Constant(F0),
                                                trainable=False)
                               )


        super(RPBinaryPooling2, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        n_frames = X.get_shape().as_list()[1]
        in_dim = float(X.get_shape().as_list()[-1])
        z_list = []
        z = 0
        for ii in range(self.n_components):

            z1 = K.dot(X, self.E_list[ii]  )
            z2 = K.dot(X, self.F_list[ii]  )

            # outer product
            z1 = self.act_fun(K.expand_dims(z1, axis=-1)) 
            z2 = self.act_fun(K.expand_dims(z2, axis=-2)) 
            z12 = tf.matmul(z1, z2)
            z12 = K.reshape(z12, [-1, n_frames, self.n_basis**2]) / in_dim

            ## use power and l2 normalization
            if self.use_normalization:
                z12 = K.sign(z12) * K.sqrt(K.abs(z12))
                z12 = K.l2_normalize(z12, axis=-1)

            z_list.append(z12)
            z += z12/self.n_components
        
        ## how to fuse the outputs from individual components
        # z = K.concatenate(z_list, axis=-1)

        

        ## pooling to reduce time sequence
        if self.out_fusion_type == 'avg':
            out_pool = AveragePooling1D(pool_size=self.stride,
                                        strides=self.stride,
                                        padding='same')(z) 
        elif self.out_fusion_type == 'max':
            out_pool = MaxPooling1D(self.stride)(z) 

        return out_pool



    def get_mat(self):
        return [self.E_list, self.F_list]


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)








class RPGaussianPooling(Layer):
    def __init__(self,
                 n_basis=8,
                 n_components=1, 
                 init_sigma=None,
                 use_normalization=False,
                 activation=None,
                 learnable_radius=True,
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        self.n_basis = n_basis
        self.out_dim = n_basis**2
        self.n_components=n_components
        self.out_fusion_type = out_fusion_type
        self.stride = stride
        self.use_normalization = use_normalization
        self.time_window_size = time_window_size
        self.learnable_radius = learnable_radius
        self.init_sigma = init_sigma

        if activation == None:
            self.act_fun = tf.keras.activations.linear
        elif activation == 'tanh':
            self.act_fun = tf.keras.activations.tanh
        elif activation == 'relu':
            self.act_fun = tf.keras.activations.relu

        super(RPGaussianPooling, self).__init__(**kwargs)



    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]

        if self.n_basis > in_dim:
           print('[ERROR]: n_basis must not be larger than in_dim! Program terminates')
           sys.exit()



        ## define the two matrix with orthogonal columns
        self.E_list = []
        self.F_list = []
        if not self.init_sigma:
            init_sigma = np.sqrt(in_dim)
        else:
            init_sigma = self.init_sigma



        for n in range(self.n_components):
            
            E0 = self.add_weight(name='E_{}'.format(n),
                                shape=[in_dim, self.n_basis], 
                                initializer=keras.initializers.Orthogonal(),
                                trainable=False)
            sigma_e = self.add_weight(name='sE_{}'.format(n),
                                shape=[1], 
                                initializer=keras.initializers.Constant(init_sigma),
                                constraint=keras.constraints.NonNeg(),
                                trainable=self.learnable_radius)
            self.E_list.append( np.sqrt(in_dim) / (K.epsilon() + sigma_e) * E0  )

            F0 = self.add_weight(name='F_{}'.format(n),
                                shape=[in_dim, self.n_basis], 
                                initializer=keras.initializers.Orthogonal(),
                                trainable=False)
            sigma_f = self.add_weight(name='sF_{}'.format(n),
                                shape=[1], 
                                initializer=keras.initializers.Constant(init_sigma),
                                constraint=keras.constraints.NonNeg(),
                                trainable=self.learnable_radius)
            self.F_list.append( np.sqrt(in_dim) / (K.epsilon() + sigma_f) * F0  )

        super(RPGaussianPooling, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        n_frames = X.get_shape().as_list()[1]
        in_dim = float(X.get_shape().as_list()[-1])
        z_list = []

        for ii in range(self.n_components):

            z1 = K.dot(X, self.E_list[ii]  )
            z2 = K.dot(X, self.F_list[ii]  )

            # outer product
            z1 = self.act_fun(K.expand_dims(z1, axis=-1)) 
            z2 = self.act_fun(K.expand_dims(z2, axis=-2)) 
            z12 = tf.matmul(z1, z2)
            z12 = K.reshape(z12, [-1, n_frames, self.n_basis**2])

            ## use power and l2 normalization
            if self.use_normalization:
                z12 = K.sign(z12) * K.sqrt(K.abs(z12))
                z12 = K.l2_normalize(z12, axis=-1)

            z_list.append(z12)

        if len(z_list) > 1:
            z = keras.layers.Average()(z_list)
        else:
            z = z_list[0]
        
        ## pooling to reduce time sequence
        if self.out_fusion_type == 'avg':
            out_pool = AveragePooling1D(pool_size=self.stride,
                                        strides=self.stride,
                                        padding='same')(z) 
        elif self.out_fusion_type == 'max':
            out_pool = MaxPooling1D(self.stride)(z) 

        return out_pool


    def get_mat(self):
        return [self.E_list, self.F_list]


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)






class RPGaussianPooling2(Layer):
    def __init__(self,
                 n_basis=8,
                 n_components=1, 
                 init_sigma=None,
                 use_normalization=False,
                 activation=None,
                 learnable_radius=True,
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        self.n_basis = n_basis
        self.n_components=n_components
        self.out_fusion_type = out_fusion_type
        self.stride = stride
        self.use_normalization = use_normalization
        self.time_window_size = time_window_size
        self.learnable_radius = True
        self.init_sigma = init_sigma
        self.out_dim = n_components*(n_basis)**2
        # print('-----------init_sigma={}-------------'.format(init_sigma))
        if activation == None:
            self.act_fun = tf.keras.activations.linear
        elif activation == 'tanh':
            self.act_fun = tf.keras.activations.tanh
        elif activation == 'relu':
            self.act_fun = tf.keras.activations.relu
        super(RPGaussianPooling2, self).__init__(**kwargs)



    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]

        if self.n_basis > in_dim:
           print('[ERROR]: n_basis must not be larger than in_dim! Program terminates')
           sys.exit()

        ## define the two matrix with orthogonal columns
        self.E_list = []
        self.F_list = []
        if not self.init_sigma:
            init_sigma = np.sqrt(in_dim)
        else:
            init_sigma = self.init_sigma


        for n in range(self.n_components):
            
            E0 = self.add_weight(name='E_{}'.format(n),
                                shape=[in_dim, self.n_basis], 
                                initializer=keras.initializers.Orthogonal(),
                                trainable=False)
            sigma_e = self.add_weight(name='sE_{}'.format(n),
                                shape=[1], 
                                initializer=keras.initializers.Constant(init_sigma),
                                constraint=keras.constraints.NonNeg(),
                                trainable=self.learnable_radius)
            
            self.E_list.append( np.sqrt(in_dim) / (K.epsilon() + sigma_e) * E0  )

            F0 = self.add_weight(name='F_{}'.format(n),
                                shape=[in_dim, self.n_basis], 
                                initializer=keras.initializers.Orthogonal(),
                                trainable=False)
            sigma_f = self.add_weight(name='sF_{}'.format(n),
                                shape=[1], 
                                initializer=keras.initializers.Constant(init_sigma),
                                constraint=keras.constraints.NonNeg(),
                                trainable=self.learnable_radius)
            self.F_list.append( np.sqrt(in_dim) / (K.epsilon() + sigma_f) * F0  )

        super(RPGaussianPooling2, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        n_frames = X.get_shape().as_list()[1]
        in_dim = float(X.get_shape().as_list()[-1])
        z_list = []

        for ii in range(self.n_components):
            z1 = K.dot(X, self.E_list[ii] )
            z2 = K.dot(X, self.F_list[ii] )


            # outer product
            z1 = K.expand_dims(z1, axis=-1)
            z2 = K.expand_dims(z2, axis=-2)
            z12 = tf.matmul(z1, z2)
            z12 = K.reshape(z12, [-1, n_frames, (self.n_basis)**2])

            ## use power and l2 normalization
            if self.use_normalization:
                z12 = K.sign(z12) * K.sqrt(K.abs(z12))
                z12 = K.l2_normalize(z12, axis=-1)

            z_list.append(z12)

        z = K.concatenate(z_list, axis=-1)
        
        ## pooling to reduce time sequence
        if self.out_fusion_type == 'avg':
            out_pool = AveragePooling1D(pool_size=self.stride,
                                        strides=self.stride,
                                        padding='same')(z) 
        elif self.out_fusion_type == 'max':
            out_pool = MaxPooling1D(self.stride)(z) 

        return out_pool


    def get_mat(self):
        return [self.E_list, self.F_list]


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)















class RPLearnable(Layer):
    def __init__(self,
                 n_basis=8,
                 n_components=1, 
                 use_normalization=False,
                 activation=None,
                 learnable_radius=True,
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        self.n_basis = n_basis
        self.n_components=n_components
        self.out_fusion_type = out_fusion_type
        self.stride = stride
        self.use_normalization = use_normalization
        self.time_window_size = time_window_size
        self.learnable_radius = learnable_radius
        self.out_dim = n_components*(n_basis)**2

        if activation == None:
            self.act_fun = tf.keras.activations.linear
        elif activation == 'tanh':
            self.act_fun = tf.keras.activations.tanh
        elif activation == 'relu':
            self.act_fun = tf.keras.activations.relu

        super(RPLearnable, self).__init__(**kwargs)


    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]

        if self.n_basis > in_dim:
           print('[ERROR]: n_basis must not be larger than in_dim! Program terminates')
           sys.exit()

        ## define the two matrix with orthogonal columns
        self.E_list = []
        self.F_list = []
        

        for n in range(self.n_components):
            E0 = self.add_weight(name='E_{}'.format(n),
                                shape=[in_dim, self.n_basis], 
                                initializer='glorot_normal',
                                trainable=True)
            
            self.E_list.append( E0 )

            F0 = self.add_weight(name='F_{}'.format(n),
                                shape=[in_dim, self.n_basis], 
                                initializer='glorot_normal',
                                trainable=True)
            
            self.F_list.append( F0 )

        super(RPLearnable, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        n_frames = X.get_shape().as_list()[1]
        in_dim = float(X.get_shape().as_list()[-1])
        z_list = []

        for ii in range(self.n_components):
            z1 = K.dot(X, self.E_list[ii] )
            z2 = K.dot(X, self.F_list[ii] )


            # outer product
            z1 = K.expand_dims(z1, axis=-1)
            z2 = K.expand_dims(z2, axis=-2)
            z12 = tf.matmul(z1, z2)
            z12 = K.reshape(z12, [-1, n_frames, (self.n_basis)**2])

            ## use power and l2 normalization
            if self.use_normalization:
                z12 = K.sign(z12) * K.sqrt(K.abs(z12))
                z12 = K.l2_normalize(z12, axis=-1)

            z_list.append(z12)

        z = K.concatenate(z_list, axis=-1)
        
        ## pooling to reduce time sequence
        if self.out_fusion_type == 'avg':
            out_pool = AveragePooling1D(pool_size=self.stride,
                                        strides=self.stride,
                                        padding='same')(z) 
        elif self.out_fusion_type == 'max':
            out_pool = MaxPooling1D(self.stride)(z) 

        return out_pool



    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)
















class MultiModalLowRankPooling(Layer):
    def __init__(self,
                 n_basis=8,
                 n_components=1,
                 use_normalization=False,
                 activation=None,
                 learnable_radius=True,
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        self.n_basis = n_basis
        self.out_dim = n_basis
        self.out_fusion_type = out_fusion_type
        self.stride = stride
        self.use_normalization = use_normalization
        self.time_window_size = time_window_size
        self.learnable_radius = learnable_radius
        self.n_components = n_components
        
        if activation == None:
            self.act_fun = tf.keras.activations.linear
        elif activation == 'tanh':
            self.act_fun = tf.keras.activations.tanh
        elif activation == 'relu':
            self.act_fun = tf.keras.activations.relu

        super(MultiModalLowRankPooling, self).__init__(**kwargs)



    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]


        ## define the two matrix with orthogonal columns
        self.E = self.add_weight(name='E',
                                shape=[in_dim, self.n_basis], 
                                initializer='glorot_normal',
                                trainable=True)
        self.F = self.add_weight(name='F',
                                shape=[in_dim, self.n_basis], 
                                initializer='glorot_normal',
                                trainable=True)
            

        super(MultiModalLowRankPooling, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        z1 = K.dot(X, self.E  )
        z2 = K.dot(X, self.F  )
        z = z1*z2
        
        ## pooling to reduce time sequence
        if self.out_fusion_type == 'avg':
            out_pool = AveragePooling1D(pool_size=self.stride,
                                        strides=self.stride,
                                        padding='same')(z) 
        elif self.out_fusion_type == 'max':
            out_pool = MaxPooling1D(self.stride)(z) 

        return out_pool


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)








class FBM(Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_factor=20,
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_factor = n_factor

        self.out_fusion_type = out_fusion_type
        self.stride = stride
        self.act_fun = tf.keras.activations.linear

        super(FBM, self).__init__(**kwargs)



    def build(self, input_shape):
        self.shape=input_shape
        in_dim = self.in_dim


        ## define the two matrix with orthogonal columns
        self.E = self.add_weight(name='E',
                                shape=[in_dim, self.n_factor, self.out_dim], 
                                initializer='glorot_normal',
                                trainable=True)
            
        self.W = self.add_weight(name='W',
                                shape=[in_dim, self.out_dim], 
                                initializer='glorot_normal',
                                trainable=True)
        
        self.b = self.add_weight(name='b',
                                shape=[1,self.out_dim], 
                                initializer='zeros',
                                trainable=True)


        super(FBM, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        first_term = K.dot(X, self.W)
        second_term = tf.einsum('btd,dkD->btD', X, self.E)
        z = self.b + first_term + second_term

        
        ## pooling to reduce time sequence
        if self.out_fusion_type == 'avg':
            out_pool = AveragePooling1D(pool_size=self.stride,
                                        strides=self.stride,
                                        padding='same')(z) 
        elif self.out_fusion_type == 'max':
            out_pool = MaxPooling1D(self.stride)(z) 

        return out_pool


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)

















from scipy.linalg import hadamard

def get_reverse_hadamard(in_dim, n_cols):
    E0 = hadamard(in_dim)
    E0_signchange = np.sum(((np.roll(E0,1,axis=0)-E0) != 0).astype(int), axis=0)
    E0 = E0[:, np.argsort(E0_signchange)[::-1]]
    
    return E0[:, :n_cols]



class ReHadamardPooling(Layer):
    def __init__(self,
                 n_basis=8,
                 n_components=1,
                 use_normalization=True,
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        self.n_basis = n_basis
        self.out_dim = n_basis**2
        self.n_components=n_components
        self.out_fusion_type = out_fusion_type
        self.stride = stride
        self.use_normalization = use_normalization
        self.time_window_size = time_window_size
      
        super(ReHadamardPooling, self).__init__(**kwargs)



    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]

        if self.n_basis > in_dim:
           print('[ERROR]: n_basis must not be larger than in_dim! Program terminates')
           sys.exit()

        if (in_dim & (in_dim - 1)) != 0:
           print('[ERROR]: input feature dimension should be power of 2')
           sys.exit()


        ## define the two matrix with orthogonal columns
        self.E_list = []
        self.F_list = []
        for n in range(self.n_components):
            E0 = get_reverse_hadamard(in_dim, self.n_basis)

            self.E_list.append( self.add_weight(name='E_{}'.format(n),
                                                shape=[in_dim, self.n_basis], 
                                                initializer=keras.initializers.Constant(E0),
                                                trainable=False)
                               )

            # F0 = np.sign(np.random.standard_normal([in_dim, self.n_basis]))
            self.F_list.append( self.add_weight(name='F_{}'.format(n),
                                                shape=[in_dim, self.n_basis], 
                                                initializer=keras.initializers.Constant(E0),
                                                trainable=False)
                               )



        super(ReHadamardPooling, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        n_frames = X.get_shape().as_list()[1]
        in_dim = float(X.get_shape().as_list()[-1])
        z_list = []

        for ii in range(self.n_components):

            z1 = K.dot(X, self.E_list[ii]  )
            z2 = K.dot(X, self.F_list[ii]  )

            # outer product
            z1 = K.expand_dims(z1, axis=-1)
            z2 = K.expand_dims(z2, axis=-2)
            z12 = tf.matmul(z1, z2)
            z12 = K.reshape(z12, [-1, n_frames, self.n_basis**2])

            ## use power and l2 normalization
            if self.use_normalization:
                z12 = K.sign(z12) * K.sqrt(K.abs(z12))
                z12 = K.l2_normalize(z12, axis=-1)

            z_list.append(z12)

        if len(z_list) > 1:
            z = keras.layers.Average()(z_list)
        else:
            z = z_list[0]
        
        ## pooling to reduce time sequence
        if self.out_fusion_type == 'avg':
            out_pool = AveragePooling1D(pool_size=self.stride,
                                        strides=self.stride,
                                        padding='same')(z) 
        elif self.out_fusion_type == 'max':
            out_pool = MaxPooling1D(self.stride)(z) 

        return out_pool


    def get_mat(self):
        return [self.E_list, self.F_list]


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)








class TensorRelaxationPooling2(Layer):
    def __init__(self,
                 n_basis,
                 use_bias=False,
                 use_normalization=False,
                 out_activation='linear',
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        self.n_basis = n_basis
        self.out_dim = n_basis**2
        self.out_fusion_type = out_fusion_type
        self.stride = stride
        self.use_bias = use_bias
        self.use_normalization = use_normalization
        self.time_window_size = time_window_size
        self.activation = out_activation
        super(TensorRelaxationPooling2, self).__init__(**kwargs)



    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]
        if self.n_basis > in_dim:
           print('[ERROR]: n_basis must not be larger than in_dim! Program terminates')
           sys.exit()


        self.E_layer = keras.layers.Conv1D(self.n_basis,
                                            kernel_size=1,
                                            activation=self.activation,
                                            use_bias=self.use_bias
                                            # kernel_constraint= non_neg_unit_norm(axis=1)
                                            # kernel_constraint=keras.constraints.NonNeg()
                                            # kernel_regularizer=regularizers.l1(100.0)
                                            )

        self.F_layer = keras.layers.Conv1D(self.n_basis,
                                            kernel_size=1,
                                            activation=self.activation,
                                            use_bias=self.use_bias
                                            # kernel_constraint= non_neg_unit_norm(axis=1)
                                            # kernel_constraint=keras.constraints.NonNeg()
                                            # kernel_regularizer=regularizers.l1(100.0)
                                            )

        super(TensorRelaxationPooling2, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        n_frames = X.get_shape().as_list()[1]
        # self.E = K.l2_normalize(self.E, axis=0)
        # self.F = K.l2_normalize(self.F, axis=0)
        # self.E_softmax = K.softmax(self.E, axis=0)
        # self.F_softmax = K.softmax(self.F, axis=0)

        z1 = self.E_layer(X)
        z2 = self.F_layer(X)

        # if self.activation == 'tanh':
        #     z1 = K.tanh(z1)
        #     z2 = K.tanh(z2)


        # outer product
        z1 = K.expand_dims(z1, axis=-1)
        z2 = K.expand_dims(z2, axis=-2)
        z = tf.matmul(z1, z2)
        z = K.reshape(z, [-1, n_frames, self.n_basis**2])
        

        ## use power and l2 normalization
        if self.use_normalization:
            z = SqrtAcfun(theta=1e-3)(z)
            z = K.l2_normalize(z, axis=-1)


        ## pooling to reduce time sequence
        if self.out_fusion_type == 'avg':
            out_pool = AveragePooling1D(pool_size=self.stride,
                                        strides=self.stride,
                                        padding='same')(z) 
        elif self.out_fusion_type == 'max':
            out_pool = MaxPooling1D(self.stride)(z) 
        elif self.out_fusion_type == 'w-sum':
            ### now out is [batch, T, out_dim], we do temporal local pooling
            #### zero padding
            out = ZeroPadding1D((self.time_window_size//2))(z)

            W = tf.reshape(self.conv_filter, [1, -1, 1]) # [1, |Nt|, 1]
            out_pool_list = [ K.sum(out[:, i:i+self.time_window_size, :]*W, axis=1)
                             for i in range(0,n_frames, self.stride)  ]

            out_pool = K.stack(out_pool_list,axis=1)
        elif self.out_fusion_type == 'linearproj':
            out_pool = Conv1D(self.out_dim, 1, strides=self.stride, padding='same')(z)


        return out_pool


    def get_weights(self):
        return [self.E_layer.get_weights()[0], self.F_layer.get_weights()[0]]

    def get_conv1d_layers(self):
        return [self.E_layer, self.F_layer]


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, self.out_dim)







class FullCorrelationLayer(Layer):
    def __init__(self,
                 out_fusion_type='avg', # or max or w-sum
                 stride=2, 
                 time_window_size=5,
                 **kwargs):
        
        
        self.out_fusion_type = out_fusion_type
        self.stride = stride
        
        super(FullCorrelationLayer, self).__init__(**kwargs)



    def build(self, input_shape):
        self.shape=input_shape
        in_dim = input_shape[-1]
        

        super(FullCorrelationLayer, self).build(input_shape)


    def call(self, X):
        ### here X is the entire mat with [batch, T, D]
        n_frames = X.get_shape().as_list()[1]
      
        # outer product
        z1 = K.expand_dims(X, axis=-1)
        z2 = K.expand_dims(X, axis=-2)
        z = tf.matmul(z1, z2)
        z = K.reshape(z, [-1, n_frames, self.shape[-1]**2])
        
        ## pooling to reduce time sequence
        if self.out_fusion_type == 'avg':
            out_pool = AveragePooling1D(pool_size=self.stride,
                                        strides=self.stride,
                                        padding='same')(z) 
        elif self.out_fusion_type == 'max':
            out_pool = MaxPooling1D(self.stride)(z) 
        elif self.out_fusion_type == 'w-sum':
            ### now out is [batch, T, out_dim], we do temporal local pooling
            #### zero padding
            out = ZeroPadding1D((self.time_window_size//2))(z)

            W = tf.reshape(self.conv_filter, [1, -1, 1]) # [1, |Nt|, 1]
            out_pool_list = [ K.sum(out[:, i:i+self.time_window_size, :]*W, axis=1)
                             for i in range(0,n_frames, self.stride)  ]

            out_pool = K.stack(out_pool_list,axis=1)
        elif self.out_fusion_type == 'linearproj':
            out_pool = Conv1D(self.out_dim, 1, strides=self.stride, padding='same')(z)


        return out_pool


    def compute_output_shape(self, input_shape):
        
        return(input_shape[0], input_shape[1]//self.stride, input_shape[-1]**2)









def constrained_loss(mat_list, weights=1.0, loss_type='orthogonal'):
    
    def loss(y_true, y_pred):
        constraint = 0

        for i in range(len(mat_list)):
            if loss_type == 'orthogonal':
                constraint0 = (K.dot(K.transpose(mat_list[i]), mat_list[i] ) - 
                                    K.eye(mat_list[i].get_shape().as_list()[1]) )**2 
                

            elif loss_type == 'softbinary':
                constraint0 = K.mean( K.abs(mat_list[i]**2 - 1.0))

            constraint += K.sum(constraint0)

        return weights*(constraint) + K.categorical_crossentropy(y_true, y_pred)

    return loss
