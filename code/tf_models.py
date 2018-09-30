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
    return Merge(mode='mul')([tanh_out, sigm_out])



def lp_normalization(x,p=2):
    if p == 2:
        return K.l2_normalize(x, axis=-1)
    else:
        norm_x = tf.maximum(tf.norm(x, ord=p, axis=-1, keepdims=True), 1e-6)
        return x/norm_x


def power_normalization(x):
    y = tf.sign(x) * tf.sqrt(tf.abs(x))
    return y





#  -------------------------------------------------------------
def temporal_convs_linear(n_nodes, conv_len, n_classes, n_feat, max_len, 
                        causal=False, loss='categorical_crossentropy', 
                        optimizer='adam', return_param_str=False):
    """ Used in paper: 
    Segmental Spatiotemporal CNNs for Fine-grained Action Segmentation
    Lea et al. ECCV 2016

    Note: Spatial dropout was not used in the original paper. 
    It tends to improve performance a little.  
    """

    inputs = Input(shape=(max_len,n_feat))
    if causal: model = ZeroPadding1D((conv_len//2,0))(model)
    model = Convolution1D(n_nodes, conv_len, input_dim=n_feat, input_length=max_len, border_mode='same', activation='relu')(inputs)
    if causal: model = Cropping1D((0,conv_len//2))(model)

    model = SpatialDropout1D(0.3)(model)

    model = TimeDistributed(Dense(n_classes, activation="softmax" ))(model)
    
    model = Model(input=inputs, output=model)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal")

    if return_param_str:
        param_str = "tConv_C{}".format(conv_len)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model






'''
low-rank approximation: backpropagation of SVD
https://github.com/tensorflow/tensorflow/issues/6503
https://gist.github.com/psycharo/60f58d5435281bdea8b9d4ee4f6e895b
'''

def mmsym(x):
    return (x + tf.transpose(x, [0,1,3,2])) / 2

def mmdiag(x):
    return tf.matrix_diag(tf.matrix_diag_part(x))


def get_eigen_K(x, square=False):
    """
    Get K = 1 / (sigma_i - sigma_j) for i != j, 0 otherwise

    Parameters
    ----------
    x : tf.Tensor with shape as [..., dim,]

    Returns
    -------

    """
    if square:
        x = tf.square(x)
    res = tf.expand_dims(x, 2) - tf.expand_dims(x, 3)
    res += tf.eye(tf.shape(res)[-1])
    res = 1 / res
    res -= tf.eye(tf.shape(res)[-1])

    # Keep the results clean
    res = tf.where(tf.is_nan(res), tf.zeros_like(res), res)
    res = tf.where(tf.is_inf(res), tf.zeros_like(res), res)
    return res

@ops.RegisterGradient('SvdGrad')
def gradient_svd(op, grad_s, grad_u, grad_v):
    """
    Define the gradient for SVD, we change it to SVD a matrix sequence
    References
        Ionescu, C., et al, Matrix Backpropagation for Deep Networks with Structured Layers
        
    Parameters
    ----------
    op
    grad_s
    grad_u
    grad_v

    Returns
    -------
    """

    s, U, V = op.outputs
    
    V_t = tf.transpose(V, [0,1,3,2])
    U_t = tf.transpose(U, [0,1,3,2])
    K = get_eigen_K(s, True)
    K_t = tf.transpose(K, [0,1,3,2]) 

    S = tf.matrix_diag(s)
    grad_S = tf.matrix_diag(grad_s)
    D = tf.matmul(grad_u, 1.0/S)
    D_t = tf.transpose(D, [0,1,3,2])


    # compose the full gradient 
    term1 = tf.matmul(D, V_t)
    
    term2_1 = mmdiag(grad_S - tf.matmul(U_t, D))
    term2 = tf.matmul(U, tf.matmul(term2_1, V_t))

    term3_1 = tf.matmul(V, tf.matmul(D_t, tf.matmul(U, S)))
    term3_2 = mmsym(K_t * tf.matmul(V_t, grad_v-term3_1))
    term3 = 2*tf.matmul(U, tf.matmul(S, tf.matmul(term3_2, V_t)))

    dL_dX = term1+term2+term3
  
    return dL_dX


def sort_tensor_column(X, col_idx):
    # X - the tensor with shape [batch, time, feature_dim, feature_dim]
    # col_idx - the column index with shape[batch, time, r]
    # this function returns a tensor with selected columns by r, i.e. return a tensor with [batch, time, feature_dim, r]
    # notice that the first dimension batch is usually None
    #n_batch = X.get_shape().as_list()[0]
    n_batch = 4
    n_time = X.get_shape().as_list()[1]
    n_dim = X.get_shape().as_list()[2]
    n_rank = col_idx.get_shape().as_list()[-1]
    Xt = tf.transpose(X, [0,1,3,2])
    Xt = tf.reshape(Xt, [n_batch*n_time, n_dim, n_dim])
    col_idx = tf.reshape(col_idx, [n_batch*n_time, n_rank])
    Xt_list = tf.unstack(Xt, axis=0)
    X_sort_list = [tf.gather_nd(Xt_list[t], col_idx[t,:]) for t in range(len(Xt_list))]
    print('X_sort_list[0].shape='+str(X_sort_list[0].shape))
    X_sort = tf.stack(X_sort_list, axis=0)
    X_sort = tf.reshape(X_sort,[n_batch, n_time, n_rank, n_dim])
    X_sort = tf.transpose(X_sort, [0,1,3,2])
    return X_sort




class EigenPooling(Layer):
    def __init__(self, rank,method='svd', **kwargs):
        self.rank = rank
        self.method = method
        super(EigenPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(EigenPooling,self).build(input_shape)


    def call(self,x):
        if self.method is 'eigen': 
        ## eigendecomposition
            e,v = tf.self_adjoint_eig(x)
            v_size = v.get_shape().as_list()
            e = tf.abs(e)
            e1,idx = tf.nn.top_k(e, k=self.rank)
            E1 = tf.sqrt(tf.matrix_diag(e1))
            u = sort_tensor_column(v, idx)

            print('v.shape='+str(v.shape))
            print('idx.shape='+str(idx.shape))
            print('u.shape='+str(u.shape))
            l = tf.matmul(u[:,:,:,:self.rank],E1[:,:,:self.rank,:self.rank]) 


	## signlar value decomposition
        elif self.method is 'svd':
            G = tf.get_default_graph()

            with G.gradient_override_map({'Svd':'SvdGrad'}):                
                s,u,v = tf.svd(x, full_matrices=True)
                l = tf.matmul(u[:,:,:,:self.rank], tf.matrix_diag(tf.sqrt(1e-5+s[:,:,:self.rank])))
        else: 
            sys.exit('[ERROR] the specified method for matrix decomposition is not valid')

        return l


    def call(self,x):
        G = tf.get_default_graph()
        
        d = x.shape[-1]
        ## eigendecomposition
        #e,v = tf.self_adjoint_eig(x)
        #e = tf.abs(e)
        #e1,idx = tf.nn.top_k(e, k=self.rank)
        #e1 = tf.matrix_diag(e1)

        #v_list = tf.unstack(v, axis=1)
        #vr_list = [tf.gather(xx, idx[]



        #print(idx)
        #print(v1.shape)
        #l = tf.matmul(v1, e1)
        #print(l.shape) 
	## signlar value decomposition
        
            # G = tf.get_default_graph()

        with G.gradient_override_map({'Svd':'SvdGrad'}):    
            
            s,u,v = tf.svd(x, full_matrices=True)
            l = tf.matmul(u[:,:,:,:self.rank], tf.matrix_diag(s[:,:,:self.rank]))
        return l
    
>>>>>>> 1f5e3da9e78a1240cc989a1c898615cc2d103e57
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2],self.rank)







def tensor_product_local(X,W):
    #input: X in [batch, T, channel]
    #input: W in [T,T]
    #compute X^T * W * X  (* is multiplication)
    n_channels = X.shape[-1]

    A = K.dot(K.permute_dimensions(X, [0,2,1]), W)
    B = K.batch_dot(A,X)
    #return K.reshape(B,[-1,n_channels*n_channels])
    return B

def weighted_average_local(X,w):
    W = K.expand_dims(w,axis=-1)
    W = K.repeat_elements(W, X.shape[-1], axis=-1)
    y = X*W

    return K.sum(y, axis=1, keepdims=False)



def tensor_product(inputs, st_conv_filter_one, conv_len, stride=1):
    # input - [batch, time, channels]
    local_size=conv_len
    n_frames = inputs.shape[1]

    x = ZeroPadding1D((local_size//2))(inputs)
    W = Lambda(lambda x: tf.diag(x))(st_conv_filter_one)

    y = [ tensor_product_local(x[:,i:i+local_size,:],W) for i in range(0,n_frames,stride) ]

    outputs =K.stack(y,axis=1) 
    outputs = K.reshape(outputs, [-1,outputs.shape[1],outputs.shape[-2]*outputs.shape[-1] ])
    return outputs




def tensor_product_with_mean(inputs, st_conv_filter_one, conv_len, feature, stride=1):
    # input - [batch, time, channels]

    if feature not in ['mean','cov','mean_cov']:
        print('[ERROR]: feature for bilinear pooling is not valid')
        sys.exit()

    local_size=conv_len
    n_frames = inputs.shape[1]
    x = ZeroPadding1D(local_size//2)(inputs)

    
    # compute mean
    mu_list = [weighted_average_local(x[:,i:i+local_size,:], st_conv_filter_one) for i in range(0,n_frames)]
    mu = K.stack(mu_list, axis=1)

    
    if feature=='mean':
        return mu[:,::stride, :]

    # compute variance
    x_centered = inputs-mu
    x_centered = ZeroPadding1D(local_size//2)(x_centered)
    W = Lambda(lambda x: tf.diag(x))(st_conv_filter_one)
    sigma_list = [ tensor_product_local(x_centered[:,i:i+local_size,:],W) for i in range(0,n_frames,stride) ]
    sigma =K.stack(sigma_list,axis=1) 
    sigma = K.reshape(sigma, [-1,sigma.shape[1], sigma.shape[-2]*sigma.shape[-1]])

    if feature == 'cov':
        return sigma

    # concatenate mean and covariance
    mu = mu[:,::stride,:]
    outputs = K.concatenate([mu, sigma])


    if feature == 'mean_cov':
        return outputs




def tensor_product_with_mean2(inputs, st_conv_filter_one, st_conv_filter_two, conv_len, feature, stride=1, rank=None, approx='eigen'):
    # input - [batch, time, channels]

    if feature not in ['mean','cov','mean_cov']:
        print('[ERROR]: feature for bilinear pooling is not valid')
        sys.exit()

    local_size=conv_len
    n_frames = inputs.shape[1]

    x = ZeroPadding1D(local_size//2)(inputs)

    
    # compute mean
    mu_list = [weighted_average_local(x[:,i:i+local_size,:], st_conv_filter_one) for i in range(0,n_frames)]
    mu = K.stack(mu_list, axis=1)

    
    if feature=='mean':
        return mu[:,::stride, :]

    # compute variance
    x_centered = inputs-mu
    x_centered = ZeroPadding1D(local_size//2)(x_centered)
    W = Lambda(lambda x: tf.diag(x))(st_conv_filter_two)
    
    sigma_list = [ tensor_product_local(x_centered[:,i:i+local_size,:],W) for i in range(0,n_frames,stride) ]
    sigma =K.stack(sigma_list,axis=1)
   

    # low rank approximation
    if rank is not None:
        if rank >= local_size:
            sys.exit('[ERROR]:the value of rank should not exceed or equal to the neighboring size')

        if approx is 'svd' or 'eigen':
        #with K.tf.device('/cpu:0'):
            sigma = EigenPooling(rank, method=approx)(sigma)
        elif approx is 'sorting':
            #todo
            print('-- this method is under construction')
        else:
            sys.exit('[ERROR]: the low rank approximation type is not valid')
    # print('sigma.shape='+str(sigma.shape))
 
    sigma = K.reshape(sigma, [-1, sigma.shape[1], sigma.shape[-2]*sigma.shape[-1]])
 
    if feature == 'cov':
        return sigma

    # concatenate mean and covariance
    mu = mu[:,::stride,:]
    outputs = K.concatenate([mu, sigma])

    print(outputs.shape)
    if feature == 'mean_cov':
        return outputs







class CharbonnierAcfun(Layer):
    def __init__(self, **kwargs):
        super(CharbonnierAcfun, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        self.gamma = self.add_weight(name='gamma',shape=[1],initializer='ones',trainable=True)
        super(CharbonnierAcfun,self).build(input_shape)

    def call(self,x):
        x = 2*self.gamma**2 * K.sqrt(1+x**2 / self.gamma )-2*self.gamma**2
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape)





class SwishAcfun(Layer):
    def __init__(self, **kwargs):
        super(SwishAcfun, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        self.gamma = self.add_weight(name='gamma',shape=[1],initializer='ones',trainable=True)
        super(SwishAcfun,self).build(input_shape)

    def call(self,x):
        x = x*K.sigmoid(self.gamma*x)
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape)






class PMAcfun(Layer):
    def __init__(self, **kwargs):
        super(PMAcfun, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        self.gamma = self.add_weight(name='gamma',shape=[1],initializer='ones',trainable=True)
        super(PMAcfun,self).build(input_shape)

    def call(self,x):
        x = self.gamma**2 * K.log(1+x**2/self.gamma**2)
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape)





class SqrtAcfun(Layer):
    def __init__(self, **kwargs):
        super(SqrtAcfun, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        self.gamma = self.add_weight(name='gamma',shape=[1],initializer=keras.initializers.Constant(value=0.01),trainable=True)
        super(SqrtAcfun,self).build(input_shape)

    def call(self,x):
        x = K.sign(x)* (K.sqrt(K.abs(x)+self.gamma)-K.sqrt(self.gamma))

        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape)









class BilinearConvLinear(Layer):
    def __init__(self, n_node, time_conv_size, **kwargs):
        self.n_node = n_node
        self.time_conv_size = time_conv_size
        
        super(BilinearConvLinear, self).__init__(**kwargs)


    def build(self, input_shape):
        self.shape=input_shape
        self.st_conv_filter_one = self.add_weight(name='conv_kernel', shape=[self.time_conv_size], 
                                                initializer='normal',
                                                trainable=True)
        self.build=True
        super(BilinearConvLinear, self).build(input_shape)

    def call(self, x):
        x = tensor_product(x, self.st_conv_filter_one, self.time_conv_size)
        x = Lambda(lambda x: lp_normalization(x, p=2))(x)
        x = Conv1D(self.n_node, 1, padding='same',activation=None)(x)

        return x

    def compute_output_shape(self, input_shape):
        return(input_shape[0], input_shape[1], self.n_node)





class BilinearPooling(Layer):
    def __init__(self, time_conv_size, stride, trainable=False, **kwargs):
        
        self.time_conv_size = time_conv_size
        self.stride = stride
        self.trainable = trainable
        super(BilinearPooling, self).__init__(**kwargs)


    def build(self, input_shape):
        self.shape=input_shape
        if not self.trainable:
            self.st_conv_filter_one = self.add_weight(name='conv_kernel', shape=[self.time_conv_size], 
                                               initializer=keras.initializers.Constant(value=1.0/self.time_conv_size),
                                               trainable=False)
        else:
            self.st_conv_filter_one = self.add_weight(name='conv_kernel', shape=[self.time_conv_size], 
                                                initializer='glorot_normal',
                                                trainable=True)
        
        super(BilinearPooling, self).build(input_shape)



    def call(self, x):
        if self.stride==1:
            x = tensor_product(x, self.st_conv_filter_one, self.time_conv_size, 1)
            x = AveragePooling1D(self.stride)(x)
            x = Lambda(lambda x: lp_normalization(x, p=2))(x)
        else:
            x = tensor_product(x, self.st_conv_filter_one, self.time_conv_size, self.stride)
            #x = Lambda(lambda x: lp_normalization(x, p=2))(x)
        
        return x

    def compute_output_shape(self, input_shape):
        return(input_shape[0], input_shape[1]//self.stride, input_shape[2]*input_shape[2])






class BilinearUpsampling(Layer):
    def __init__(self, stride, **kwargs):
        self.stride = stride
        super(BilinearUpsampling, self).__init__(**kwargs)


    def build(self, input_shape):
        self.shape=input_shape
        self.st_conv_filter_one = self.add_weight(name='conv_kernel', shape=[1], 
                                                initializer='ones',
                                                trainable=False)

        super(BilinearUpsampling, self).build(input_shape)


    def call(self, x):
        x = tensor_product(x, self.st_conv_filter_one, 1)
        x = Lambda(lambda x: lp_normalization(x, p=2))(x)
        x = UpSampling1D(self.stride)(x)

        return x

    def compute_output_shape(self, input_shape):
        return(input_shape[0], input_shape[1]*self.stride, input_shape[2]*input_shape[2])






class CenteredBilinearPooling(Layer):
    def __init__(self, time_conv_size, stride, trainable=False, feature='mean_cov',**kwargs):
        
        self.time_conv_size = time_conv_size
        self.stride = stride
        self.trainable=trainable
        self.feature=feature
        super(CenteredBilinearPooling, self).__init__(**kwargs)


    def build(self, input_shape):
        self.shape=input_shape
        if not self.trainable:
            self.st_conv_filter_one = self.add_weight(name='weights', shape=[self.time_conv_size], 
                                   initializer=keras.initializers.Constant(value=1.0/self.time_conv_size),
                                   trainable=False)

        else:
            self.st_conv_filter_one = self.add_weight(name='weights', shape=[self.time_conv_size], 
                                                initializer='glorot_normal',
                                                trainable=True)
        

        super(CenteredBilinearPooling, self).build(input_shape)



    def call(self, x):
        if self.stride==1:
            x = BilinearPooling(1,2, trainable=self.trainable)(x)
            # mu = x*self.time_conv_size
            # x = tensor_product(x, self.st_conv_filter_one, self.time_conv_size, 1)
            # x = AveragePooling1D(self.stride)(x)
            # x = Lambda(lambda x: lp_normalization(x, p=2))(x)

        else:


            x = tensor_product_with_mean(x, self.st_conv_filter_one, self.time_conv_size, 
                                        self.feature, self.stride)
            x = Lambda(lambda x: lp_normalization(x, p=2))(x)
        
        return x

    def compute_output_shape(self, input_shape):
        if self.feature == 'mean_cov':
            return(input_shape[0], input_shape[1]//self.stride, (1+input_shape[2])*input_shape[2])
        elif self.feature == 'mean':
            return (input_shape[0], input_shape[1]//self.stride, input_shape[2])
        elif self.feature == 'cov':
            return(input_shape[0], input_shape[1]//self.stride, input_shape[2]*input_shape[2])
        else:
            print('[ERROR]: feature for bilinear pooling is not valid')
            sys.exit()




class CenteredBilinearPooling2(Layer):
    ## mean and cov are with different parameters
    def __init__(self, time_conv_size, stride, trainable=False, feature='mean_cov',**kwargs):
        
        self.time_conv_size = time_conv_size
        self.stride = stride
        self.trainable=trainable
        self.feature=feature
        super(CenteredBilinearPooling2, self).__init__(**kwargs)


    def build(self, input_shape):
        self.shape=input_shape
        if not self.trainable:
            self.st_conv_filter_one = self.add_weight(name='weights', shape=[self.time_conv_size], 
                                   initializer=keras.initializers.Constant(value=1.0/self.time_conv_size),
                                   trainable=False)
            self.st_conv_filter_two = self.st_conv_filter_one
        else:
            self.st_conv_filter_one = self.add_weight(name='weights', shape=[self.time_conv_size], 
                                                initializer='glorot_normal',
                                                trainable=True)
	    
            self.st_conv_filter_two = self.add_weight(name='weights2', shape=[self.time_conv_size], 
                                                initializer='glorot_normal',
                                                trainable=True)
	    

        super(CenteredBilinearPooling2, self).build(input_shape)



    def call(self, x):
        if self.stride==1:
            x = BilinearPooling(1,2, trainable=self.trainable)(x)
            # mu = x*self.time_conv_size
            # x = tensor_product(x, self.st_conv_filter_one, self.time_conv_size, 1)
            # x = AveragePooling1D(self.stride)(x)
            # x = Lambda(lambda x: lp_normalization(x, p=2))(x)

        else:


            x = tensor_product_with_mean2(x, self.st_conv_filter_one, self.st_conv_filter_two, self.time_conv_size, 
                                        self.feature, self.stride)
            # x = Lambda(lambda x: lp_normalization(x, p=1))(x)
        
        return x

    def compute_output_shape(self, input_shape):
        if self.feature == 'mean_cov':
            return(input_shape[0], input_shape[1]//self.stride, (1+input_shape[2])*input_shape[2])
        elif self.feature == 'mean':
            return (input_shape[0], input_shape[1]//self.stride, input_shape[2])
        elif self.feature == 'cov':
            return(input_shape[0], input_shape[1]//self.stride, input_shape[2]*input_shape[2])
        else:
            print('[ERROR]: feature for bilinear pooling is not valid')
            sys.exit()








class CenteredBilinearPoolingLowRank(Layer):
    ## low rank approximation of the Cov matrix
    def __init__(self, time_conv_size, stride, trainable=False, feature='mean_cov', rank=1, **kwargs):
        
        self.time_conv_size = time_conv_size
        self.stride = stride
        self.trainable=trainable
        self.feature=feature
        self.rank = rank
        super(CenteredBilinearPoolingLowRank, self).__init__(**kwargs)


    def build(self, input_shape):
        self.shape=input_shape
        if not self.trainable:
            self.st_conv_filter_one = self.add_weight(name='weights', shape=[self.time_conv_size], 
                                   initializer=keras.initializers.Constant(value=1.0/self.time_conv_size),
                                   trainable=False)
            self.st_conv_filter_two = self.st_conv_filter_one
        else:
            self.st_conv_filter_one = self.add_weight(name='weights', shape=[self.time_conv_size], 
                                                initializer='glorot_normal',
                                                trainable=True)
        
            self.st_conv_filter_two = self.add_weight(name='weights2', shape=[self.time_conv_size], 
                                                initializer='glorot_normal',
                                                trainable=True)

        super(CenteredBilinearPoolingLowRank, self).build(input_shape)



    def call(self, x):
        if self.stride==1:
            x = BilinearPooling(1,2, trainable=self.trainable)(x)
            # mu = x*self.time_conv_size
            # x = tensor_product(x, self.st_conv_filter_one, self.time_conv_size, 1)
            # x = AveragePooling1D(self.stride)(x)
            # x = Lambda(lambda x: lp_normalization(x, p=2))(x)

        else:
            x = tensor_product_with_mean2(x, self.st_conv_filter_one, self.st_conv_filter_two, self.time_conv_size, 
                                        self.feature, self.stride, rank=self.rank)
            # x = Lambda(lambda x: lp_normalization(x, p=2))(x)
        
        return x

    def compute_output_shape(self, input_shape):
        if self.feature == 'mean_cov':
            return(input_shape[0], input_shape[1]//self.stride, (1+self.rank)*input_shape[2])
        elif self.feature == 'mean':
            return (input_shape[0], input_shape[1]//self.stride, input_shape[2])
        elif self.feature == 'cov':
            return(input_shape[0], input_shape[1]//self.stride, self.rank*input_shape[2])
        else:
            print('[ERROR]: feature for bilinear pooling is not valid')
            sys.exit()










def ED_Bilinear(n_nodes, conv_len, n_classes, n_feat, max_len, 
            causal=False,
            activation='norm_relu',
            return_param_str=False, batch_size = 4, dropout_ratio=0.3):
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len,n_feat))
    model = inputs
    model = Lambda(lambda x: tf.cast(x, dtype=tf.float32))(model)
    #u_connection = False
    #branch = []

    # ---- Encoder ----
    for i in range(n_layers):


        #if u_connection:
        #    branch.append(model)


        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = Conv1D(n_nodes[i], conv_len, padding='same')(model)
        # model = BilinearConvLinear(n_nodes[i], conv_len)(model)
        if causal: model = Cropping1D((0,conv_len//2))(model)
        
        # model = Lambda(lambda x: lp_normalization(x, p=2))(model)

        if dropout_ratio != 0:
            model = SpatialDropout1D(dropout_ratio)(model)
        
        if activation=='norm_relu': 
            model = Activation('relu')(model)            
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        elif activation=='charbonnier':
            model = CharbonnierAcfun()(model)
        elif activation=='swish':
            model = SwishAcfun()(model)
        elif activation=='leaky_relu':
            model = keras.layers.LeakyReLU(alpha=0.2)(model)
        elif activation == 'pm':
            model = PMAcfun()(model)
        elif activation == 'sqrt':
            model = SqrtAcfun()(model)
        else:
            model = Activation(activation)(model)    

        
        #model = Lambda(lambda x: lp_normalization(x,p=2))(model)
        # model = MaxPooling1D(2)(model)	
        # model = BilinearPooling(5,stride=2, trainable=True)(model)
        model = CenteredBilinearPooling2(5,stride=2, trainable=True, feature='mean_cov')(model)
        # model = CenteredBilinearPoolingLowRank(25,stride=2, trainable=True, feature='mean_cov', rank=1)(model)
        
        
    # ---- Decoder ----
    for i in range(n_layers):
        model = UpSampling1D(2)(model)
        # model = BilinearUpsampling(2)(model)
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = Conv1D(n_nodes[-i-1], conv_len, padding='same')(model)
        # model = BilinearConvLinear(n_nodes[-i-1], conv_len)(model)

        if causal: model = Cropping1D((0,conv_len//2))(model)

        if dropout_ratio != 0:
            model = SpatialDropout1D(dropout_ratio)(model)

        if activation=='norm_relu': 
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        elif activation=='charbonnier':
            model = CharbonnierAcfun()(model)
        elif activation=='swish':
            model=SwishAcfun()(model)
        elif activation=='leaky_relu':
            model = keras.layers.LeakyReLU(alpha=0.2)(model)
        elif activation == 'pm':
            model = PMAcfun()(model)
        elif activation == 'sqrt':
            model = SqrtAcfun()(model)

        else:
            model = Activation(activation)(model)

        # if u_connection:
        #    model = keras.layers.Concatenate(axis=-1)([model, branch[-i-1]])
        
        print(model.shape)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation='softmax'))(model)
    model = Model(input=inputs, output=model)


    # optimizer = keras.optimizers.RMSprop(lr=0.01)
    optimizer = keras.optimizers.Adam(lr=0.01, decay=0.0)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, sample_weight_mode="temporal", metrics=['accuracy'])

    if return_param_str:
        param_str = "ED-Bilinear_C{}_L{}".format(conv_len, n_layers)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model







def ED_TCN(n_nodes, conv_len, n_classes, n_feat, max_len, 
            loss='categorical_crossentropy', causal=False, 
            optimizer="rmsprop", activation='norm_relu',
            return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len,n_feat))
    model = inputs
    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = Convolution1D(n_nodes[i], conv_len, border_mode='same')(model)
        if causal: model = Cropping1D((0,conv_len//2))(model)

        model = SpatialDropout1D(0.3)(model)
        
        if activation=='norm_relu': 
            model = Activation('relu')(model)            
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 

        else:
            model = Activation(activation)(model)    

        
        model = MaxPooling1D(2)(model)

    # ---- Decoder ----
    for i in range(n_layers):
        model = UpSampling1D(2)(model)
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = Convolution1D(n_nodes[-i-1], conv_len, border_mode='same')(model)
        if causal: model = Cropping1D((0,conv_len//2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation=='norm_relu': 
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        else:
            model = Activation(activation)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax" ))(model)
    
    model = Model(input=inputs, output=model)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal", metrics=['accuracy'])

    if return_param_str:
        param_str = "ED-TCN_C{}_L{}".format(conv_len, n_layers)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model










def ED_TCN_atrous(n_nodes, conv_len, n_classes, n_feat, max_len, 
                loss='categorical_crossentropy', causal=False, 
                optimizer="rmsprop", activation='norm_relu',
                return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(None,n_feat))
    model = inputs

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = AtrousConvolution1D(n_nodes[i], conv_len, atrous_rate=i+1, border_mode='same')(model)
        if causal: model = Cropping1D((0,conv_len//2))(model)

        model = SpatialDropout1D(0.3)(model)
        
        if activation=='norm_relu': 
            model = Activation('relu')(model)            
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        else:
            model = Activation(activation)(model)            

    # ---- Decoder ----
    for i in range(n_layers):
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = AtrousConvolution1D(n_nodes[-i-1], conv_len, atrous_rate=n_layers-i, border_mode='same')(model)      
        if causal: model = Cropping1D((0,conv_len//2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation=='norm_relu': 
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        else:
            model = Activation(activation)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax" ))(model)

    model = Model(input=inputs, output=model)

    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal", metrics=['accuracy'])

    if return_param_str:
        param_str = "ED-TCNa_C{}_L{}".format(conv_len, n_layers)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model



def TimeDelayNeuralNetwork(n_nodes, conv_len, n_classes, n_feat, max_len, 
                loss='categorical_crossentropy', causal=False, 
                optimizer="rmsprop", activation='sigmoid',
                return_param_str=False):
    # Time-delay neural network
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len,n_feat))
    model = inputs
    inputs_mask = Input(shape=(max_len,1))
    model_masks = [inputs_mask]

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = AtrousConvolution1D(n_nodes[i], conv_len, atrous_rate=i+1, border_mode='same')(model)
        # model = SpatialDropout1D(0.3)(model)
        if causal: model = Cropping1D((0,conv_len//2))(model)
        
        if activation=='norm_relu': 
            model = Activation('relu')(model)            
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        else:
            model = Activation(activation)(model)            

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(input=inputs, output=model)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal", metrics=['accuracy'])

    if return_param_str:
        param_str = "TDN_C{}".format(conv_len)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model



def Dilated_TCN(num_feat, num_classes, nb_filters, dilation_depth, nb_stacks, max_len, 
            activation="wavenet", tail_conv=1, use_skip_connections=True, causal=False, 
            optimizer='adam', return_param_str=False):
    """
    dilation_depth : number of layers per stack
    nb_stacks : number of stacks.
    """

    def residual_block(x, s, i, activation):
        original_x = x

        if causal:
            x = ZeroPadding1D(((2**i)//2,0))(x)
            conv = AtrousConvolution1D(nb_filters, 2, atrous_rate=2**i, border_mode='same',
                                        name='dilated_conv_%d_tanh_s%d' % (2**i, s))(x)
            conv = Cropping1D((0,(2**i)//2))(conv)
        else:
            conv = AtrousConvolution1D(nb_filters, 3, atrous_rate=2**i, border_mode='same',
                                    name='dilated_conv_%d_tanh_s%d' % (2**i, s))(x)                                        

        conv = SpatialDropout1D(0.3)(conv)
        # x = WaveNet_activation(conv)

        if activation=='norm_relu': 
            x = Activation('relu')(conv)
            x = Lambda(channel_normalization)(x)
        elif activation=='wavenet': 
            x = WaveNet_activation(conv) 
        else:
            x = Activation(activation)(conv)        

        #res_x  = Convolution1D(nb_filters, 1, border_mode='same')(x)
        #skip_x = Convolution1D(nb_filters, 1, border_mode='same')(x)
        x  = Convolution1D(nb_filters, 1, border_mode='same')(x)

        res_x = Merge(mode='sum')([original_x, x])

        #return res_x, skip_x
        return res_x, x

    input_layer = Input(shape=(max_len, num_feat))

    skip_connections = []

    x = input_layer
    if causal:
        x = ZeroPadding1D((1,0))(x)
        x = Convolution1D(nb_filters, 2, border_mode='same', name='initial_conv')(x)
        x = Cropping1D((0,1))(x)
    else:
        x = Convolution1D(nb_filters, 3, border_mode='same', name='initial_conv')(x)    

    for s in range(nb_stacks):
        for i in range(0, dilation_depth+1):
            x, skip_out = residual_block(x, s, i, activation)
            skip_connections.append(skip_out)

    if use_skip_connections:
        x = Merge(mode='sum')(skip_connections)
    x = Activation('relu')(x)
    x = Convolution1D(nb_filters, tail_conv, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution1D(num_classes, tail_conv, border_mode='same')(x)
    x = Activation('softmax', name='output_softmax')(x)

    model = Model(input_layer, x)
    model.compile(optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal')

    if return_param_str:
        param_str = "D-TCN_C{}_B{}_L{}".format(2, nb_stacks, dilation_depth)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model

def BidirLSTM(n_nodes, n_classes, n_feat, max_len=None, 
                causal=True, loss='categorical_crossentropy', optimizer="adam",
                return_param_str=False):
    
    inputs = Input(shape=(None,n_feat))
    model = LSTM(n_nodes, return_sequences=True)(inputs)

    # Birdirectional LSTM
    if not causal:
        model_backwards = LSTM(n_nodes, return_sequences=True, go_backwards=True)(inputs)
        model = Merge(mode="concat")([model, model_backwards])

    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)
    
    model = Model(input=inputs, output=model)
    model.compile(optimizer=optimizer, loss=loss, sample_weight_mode="temporal", metrics=['accuracy'])
    
    if return_param_str:
        param_str = "LSTM_N{}".format(n_nodes)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model

